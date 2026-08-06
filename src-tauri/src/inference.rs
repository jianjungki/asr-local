use anyhow::{anyhow, Result};
use rubato::{FastFixedIn, PolynomialDegree, Resampler};
use serde::Serialize;
use sherpa_onnx::{
    OfflinePunctuation, OfflinePunctuationConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig, OfflineZipformerCtcModelConfig, OnlineParaformerModelConfig,
    OnlineRecognizer, OnlineRecognizerConfig, OnlineStream, SileroVadModelConfig, VadModelConfig,
    VoiceActivityDetector,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

pub const SENSE_VOICE_MODEL_ID: &str = "sense-voice";
pub const PARAFORMER_STREAMING_MODEL_ID: &str = "paraformer-streaming";
pub const ZIPFORMER_ZH_MODEL_ID: &str = "zipformer-zh";
pub const DEFAULT_LOCAL_SPEECH_MODEL_ID: &str = PARAFORMER_STREAMING_MODEL_ID;
pub const DEFAULT_SENSE_VOICE_LANGUAGE: &str = "auto";

const TARGET_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_CHUNK_SAMPLES: usize = TARGET_SAMPLE_RATE as usize;
const MAX_BUFFER_SECONDS: usize = 8;
const VAD_BUFFER_SECONDS: f32 = 30.0;
const VAD_THRESHOLD: f32 = 0.5;
// A slightly longer endpoint window gives the punctuation model enough clause
// context to place commas instead of receiving many very short fragments.
const VAD_MIN_SILENCE_SECONDS: f32 = 0.45;
const VAD_MIN_SPEECH_SECONDS: f32 = 0.2;
const VAD_MAX_SPEECH_SECONDS: f32 = 12.0;
const VAD_WINDOW_SIZE: i32 = 512;
const SPEECH_PREROLL_SAMPLES: usize = 5_120;
const SENSE_VOICE_PARTIAL_INTERVAL_SAMPLES: usize = 20_480;
const ZIPFORMER_PARTIAL_INTERVAL_SAMPLES: usize = 12_288;

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalTranscriptResult {
    pub text: String,
    pub is_final: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalSpeechModelKind {
    SenseVoice,
    ParaformerStreaming,
    ZipformerZh,
}

impl LocalSpeechModelKind {
    pub fn from_id(model_id: &str) -> Result<Self> {
        match model_id.trim() {
            "" | PARAFORMER_STREAMING_MODEL_ID => Ok(Self::ParaformerStreaming),
            SENSE_VOICE_MODEL_ID => Ok(Self::SenseVoice),
            ZIPFORMER_ZH_MODEL_ID => Ok(Self::ZipformerZh),
            other => Err(anyhow!("Unsupported local speech model: {other}")),
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::SenseVoice => SENSE_VOICE_MODEL_ID,
            Self::ParaformerStreaming => PARAFORMER_STREAMING_MODEL_ID,
            Self::ZipformerZh => ZIPFORMER_ZH_MODEL_ID,
        }
    }
}

fn first_existing_model(model_dir: &Path, candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(|file_name| model_dir.join(file_name))
        .find(|path| model_file_is_ready(path))
}

fn model_file_is_ready(path: &Path) -> bool {
    path.metadata()
        .map(|metadata| metadata.is_file() && metadata.len() > 1024 * 1024)
        .unwrap_or(false)
}

pub fn vad_model_file_is_ready(path: &Path) -> bool {
    path.metadata()
        .map(|metadata| metadata.is_file() && metadata.len() > 512 * 1024)
        .unwrap_or(false)
}

pub fn punctuation_model_file_is_ready(path: &Path) -> bool {
    path.metadata()
        .map(|metadata| metadata.is_file() && metadata.len() > 50 * 1024 * 1024)
        .unwrap_or(false)
}

fn clean_transcript_text(text: &str) -> String {
    text.replace("<unk>", "")
        .replace("<UNK>", "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn ensure_terminal_punctuation(text: String) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let prefix = trimmed.trim_end_matches(|character| {
        matches!(
            character,
            '\"' | '\'' | ')' | ']' | '}' | '”' | '’' | '）' | '】' | '》'
        )
    });
    let has_punctuation = prefix.chars().last().is_some_and(|character| {
        matches!(
            character,
            '.' | '!' | '?' | ',' | ';' | ':' | '。' | '！' | '？' | '，' | '；' | '：' | '、'
        )
    });
    if has_punctuation {
        return trimmed.to_string();
    }

    let suffix = &trimmed[prefix.len()..];
    let terminal = if prefix
        .chars()
        .any(|character| matches!(character, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}'))
    {
        '。'
    } else {
        '.'
    };
    format!("{prefix}{terminal}{suffix}")
}

fn join_transcript_parts(parts: Vec<String>) -> String {
    let mut result = String::new();
    for part in parts {
        if result
            .chars()
            .last()
            .is_some_and(|character| character.is_ascii_alphanumeric())
            && part
                .chars()
                .next()
                .is_some_and(|character| character.is_ascii_alphanumeric())
        {
            result.push(' ');
        }
        result.push_str(&part);
    }
    result
}

pub fn normalize_sense_voice_language(language: Option<&str>) -> Result<String> {
    let language = language
        .unwrap_or(DEFAULT_SENSE_VOICE_LANGUAGE)
        .trim()
        .to_ascii_lowercase();
    let normalized = match language.as_str() {
        "" | "auto" => DEFAULT_SENSE_VOICE_LANGUAGE,
        "zh" | "zh-cn" => "zh",
        "yue" => "yue",
        "en" => "en",
        "ja" => "ja",
        "ko" => "ko",
        other => return Err(anyhow!("Unsupported SenseVoice language: {other}")),
    };
    Ok(normalized.to_string())
}

pub fn model_files_exist(model_dir: &Path, model_id: &str) -> bool {
    let Ok(kind) = LocalSpeechModelKind::from_id(model_id) else {
        return false;
    };
    if !model_dir.join("tokens.txt").exists() {
        return false;
    }

    match kind {
        LocalSpeechModelKind::SenseVoice => model_file_is_ready(&model_dir.join("model.int8.onnx")),
        LocalSpeechModelKind::ParaformerStreaming => {
            model_file_is_ready(&model_dir.join("encoder.int8.onnx"))
                && model_file_is_ready(&model_dir.join("decoder.int8.onnx"))
        }
        LocalSpeechModelKind::ZipformerZh => {
            first_existing_model(model_dir, &["model.int8.onnx", "model.onnx"]).is_some()
        }
    }
}

pub struct SherpaOfflineModel {
    model_id: String,
    recognition_language: String,
    recognizer: Option<OfflineRecognizer>,
    online_recognizer: Option<OnlineRecognizer>,
    online_stream: Option<OnlineStream>,
    online_last_text: String,
    vad: Option<VoiceActivityDetector>,
    punctuator: Option<OfflinePunctuation>,
    resampler: Option<FastFixedIn<f32>>,
    audio_buffer: Vec<f32>,
    speech_audio: Vec<f32>,
    speech_preroll: Vec<f32>,
    samples_at_last_partial: usize,
    speech_active: bool,
}

impl SherpaOfflineModel {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        model_id: &str,
        sense_voice_language: Option<&str>,
        vad_model_path: Option<&Path>,
        punctuation_model_path: Option<&Path>,
        sample_rate: u32,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let kind = LocalSpeechModelKind::from_id(model_id)?;
        let recognition_language = match kind {
            LocalSpeechModelKind::SenseVoice => {
                normalize_sense_voice_language(sense_voice_language)?
            }
            LocalSpeechModelKind::ParaformerStreaming => "zh-en".to_string(),
            LocalSpeechModelKind::ZipformerZh => "zh".to_string(),
        };
        let tokens_path = model_dir.join("tokens.txt");
        if !tokens_path.exists() {
            return Err(anyhow!(
                "Sherpa-ONNX tokens file not found at: {}",
                tokens_path.display()
            ));
        }

        let (recognizer, online_recognizer, online_stream) = match kind {
            LocalSpeechModelKind::ParaformerStreaming => {
                let encoder_path = model_dir.join("encoder.int8.onnx");
                let decoder_path = model_dir.join("decoder.int8.onnx");
                if !model_file_is_ready(&encoder_path) || !model_file_is_ready(&decoder_path) {
                    return Err(anyhow!(
                        "Paraformer streaming model files not found in: {}",
                        model_dir.display()
                    ));
                }

                let mut config = OnlineRecognizerConfig::default();
                config.model_config.paraformer = OnlineParaformerModelConfig {
                    encoder: Some(encoder_path.display().to_string()),
                    decoder: Some(decoder_path.display().to_string()),
                };
                config.model_config.tokens = Some(tokens_path.display().to_string());
                config.model_config.num_threads = 2;
                config.model_config.provider = Some("cpu".to_string());
                config.decoding_method = Some("greedy_search".to_string());
                // Silero VAD owns endpointing so there is only one sentence boundary source.
                config.enable_endpoint = false;

                let online_recognizer = OnlineRecognizer::create(&config)
                    .ok_or_else(|| anyhow!("Failed to create Paraformer online recognizer"))?;
                let online_stream = online_recognizer.create_stream();
                (None, Some(online_recognizer), Some(online_stream))
            }
            LocalSpeechModelKind::SenseVoice | LocalSpeechModelKind::ZipformerZh => {
                let mut config = OfflineRecognizerConfig::default();
                if kind == LocalSpeechModelKind::SenseVoice {
                    let model_path = model_dir.join("model.int8.onnx");
                    if !model_file_is_ready(&model_path) {
                        return Err(anyhow!(
                            "SenseVoice model file not found at: {}",
                            model_path.display()
                        ));
                    }
                    config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
                        model: Some(model_path.display().to_string()),
                        language: Some(recognition_language.clone()),
                        use_itn: true,
                    };
                } else {
                    let model_path =
                        first_existing_model(model_dir, &["model.int8.onnx", "model.onnx"])
                            .ok_or_else(|| {
                                anyhow!(
                                    "Zipformer Chinese model file not found in: {}",
                                    model_dir.display()
                                )
                            })?;
                    config.model_config.zipformer_ctc = OfflineZipformerCtcModelConfig {
                        model: Some(model_path.display().to_string()),
                    };
                    config.decoding_method = Some("greedy_search".to_string());
                }
                config.model_config.tokens = Some(tokens_path.display().to_string());
                config.model_config.num_threads = if kind == LocalSpeechModelKind::SenseVoice {
                    std::thread::available_parallelism()
                        .map(|count| count.get().clamp(2, 4) as i32)
                        .unwrap_or(2)
                } else {
                    2
                };
                config.model_config.provider = Some("cpu".to_string());

                let recognizer = OfflineRecognizer::create(&config)
                    .ok_or_else(|| anyhow!("Failed to create Sherpa-ONNX offline recognizer"))?;
                (Some(recognizer), None, None)
            }
        };

        let vad_model_path = vad_model_path
            .ok_or_else(|| anyhow!("Silero VAD model path is required for local recognition"))?;
        if !vad_model_file_is_ready(vad_model_path) {
            return Err(anyhow!(
                "Silero VAD model file not found at: {}",
                vad_model_path.display()
            ));
        }
        let vad_config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(vad_model_path.display().to_string()),
                threshold: VAD_THRESHOLD,
                min_silence_duration: VAD_MIN_SILENCE_SECONDS,
                min_speech_duration: VAD_MIN_SPEECH_SECONDS,
                window_size: VAD_WINDOW_SIZE,
                max_speech_duration: VAD_MAX_SPEECH_SECONDS,
            },
            sample_rate: TARGET_SAMPLE_RATE as i32,
            num_threads: 1,
            provider: Some("cpu".to_string()),
            debug: false,
            ..Default::default()
        };
        let vad = Some(
            VoiceActivityDetector::create(&vad_config, VAD_BUFFER_SECONDS)
                .ok_or_else(|| anyhow!("Failed to create Silero VAD"))?,
        );

        let punctuator = match kind {
            LocalSpeechModelKind::ParaformerStreaming | LocalSpeechModelKind::ZipformerZh => {
                let punctuation_model_path = punctuation_model_path
                    .ok_or_else(|| anyhow!("Punctuation model path is required"))?;
                if !punctuation_model_file_is_ready(punctuation_model_path) {
                    return Err(anyhow!(
                        "Punctuation model file not found at: {}",
                        punctuation_model_path.display()
                    ));
                }
                let mut punctuation_config = OfflinePunctuationConfig::default();
                punctuation_config.model.ct_transformer =
                    Some(punctuation_model_path.display().to_string());
                punctuation_config.model.num_threads = 1;
                Some(
                    OfflinePunctuation::create(&punctuation_config)
                        .ok_or_else(|| anyhow!("Failed to create punctuation model"))?,
                )
            }
            LocalSpeechModelKind::SenseVoice => None,
        };

        let resampler = if sample_rate != TARGET_SAMPLE_RATE {
            let ratio = TARGET_SAMPLE_RATE as f64 / sample_rate as f64;
            Some(FastFixedIn::new(
                ratio,
                2.0,
                PolynomialDegree::Cubic,
                1024,
                1,
            )?)
        } else {
            None
        };

        Ok(Self {
            model_id: kind.id().to_string(),
            recognition_language,
            recognizer,
            online_recognizer,
            online_stream,
            online_last_text: String::new(),
            vad,
            punctuator,
            resampler,
            audio_buffer: Vec::new(),
            speech_audio: Vec::new(),
            speech_preroll: Vec::new(),
            samples_at_last_partial: 0,
            speech_active: false,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn recognition_language(&self) -> &str {
        &self.recognition_language
    }

    pub fn reset_session(&mut self) {
        self.audio_buffer.clear();
        self.speech_audio.clear();
        self.speech_preroll.clear();
        self.samples_at_last_partial = 0;
        self.speech_active = false;
        self.online_last_text.clear();
        if let Some(recognizer) = &self.online_recognizer {
            self.online_stream = Some(recognizer.create_stream());
        }
        if let Some(vad) = &self.vad {
            vad.clear();
            vad.reset();
        }
    }

    fn decode_audio(&self, samples: &[f32]) -> String {
        if samples.is_empty() {
            return String::new();
        }

        let recognizer = self
            .recognizer
            .as_ref()
            .expect("offline decoder is only used by offline models");
        let stream = recognizer.create_stream();
        stream.accept_waveform(TARGET_SAMPLE_RATE as i32, samples);
        recognizer.decode(&stream);

        stream
            .get_result()
            .map(|result| result.text.trim().to_string())
            .map(|text| clean_transcript_text(&text))
            .unwrap_or_default()
    }

    fn decode_paraformer_chunk(&mut self, samples: &[f32]) -> Result<String> {
        let recognizer = self
            .online_recognizer
            .as_ref()
            .ok_or_else(|| anyhow!("Paraformer online recognizer is not initialized"))?;
        let stream = self
            .online_stream
            .as_ref()
            .ok_or_else(|| anyhow!("Paraformer online stream is not initialized"))?;

        stream.accept_waveform(TARGET_SAMPLE_RATE as i32, samples);
        while recognizer.is_ready(stream) {
            recognizer.decode(stream);
        }

        Ok(recognizer
            .get_result(stream)
            .map(|result| clean_transcript_text(result.text.trim()))
            .unwrap_or_default())
    }

    fn finish_paraformer_stream(&mut self) -> Result<String> {
        let final_hypothesis = {
            let recognizer = self
                .online_recognizer
                .as_ref()
                .ok_or_else(|| anyhow!("Paraformer online recognizer is not initialized"))?;
            let stream = self
                .online_stream
                .as_ref()
                .ok_or_else(|| anyhow!("Paraformer online stream is not initialized"))?;

            stream.input_finished();
            while recognizer.is_ready(stream) {
                recognizer.decode(stream);
            }
            recognizer
                .get_result(stream)
                .map(|result| clean_transcript_text(result.text.trim()))
                .unwrap_or_default()
        };

        let raw_text = if final_hypothesis.is_empty() {
            self.online_last_text.clone()
        } else {
            final_hypothesis
        };
        let next_stream = self
            .online_recognizer
            .as_ref()
            .ok_or_else(|| anyhow!("Paraformer online recognizer is not initialized"))?
            .create_stream();
        self.online_stream = Some(next_stream);
        self.online_last_text.clear();
        Ok(self.finalize_text(raw_text))
    }

    fn finalize_text(&self, text: String) -> String {
        if text.is_empty() {
            return text;
        }
        let punctuated = self
            .punctuator
            .as_ref()
            .and_then(|punctuator| punctuator.add_punctuation(&text))
            .filter(|punctuated| !punctuated.trim().is_empty())
            .unwrap_or(text);
        ensure_terminal_punctuation(punctuated)
    }

    fn take_ready_vad_segments(&self) -> Vec<Vec<f32>> {
        let Some(vad) = &self.vad else {
            return Vec::new();
        };

        let mut segments = Vec::new();
        while let Some(segment) = vad.front() {
            segments.push(segment.samples().to_vec());
            drop(segment);
            vad.pop();
        }
        segments
    }

    fn decode_final_segments(&self, segments: Vec<Vec<f32>>) -> String {
        let parts = segments
            .iter()
            .filter_map(|samples| {
                let text = self.finalize_text(self.decode_audio(samples));
                (!text.is_empty()).then_some(text)
            })
            .collect::<Vec<_>>();
        join_transcript_parts(parts)
    }

    fn update_preroll(&mut self, samples: &[f32]) {
        self.speech_preroll.extend_from_slice(samples);
        if self.speech_preroll.len() > SPEECH_PREROLL_SAMPLES {
            let excess = self.speech_preroll.len() - SPEECH_PREROLL_SAMPLES;
            self.speech_preroll.drain(..excess);
        }
    }

    fn partial_interval_samples(&self) -> usize {
        if self.model_id == SENSE_VOICE_MODEL_ID {
            SENSE_VOICE_PARTIAL_INTERVAL_SAMPLES
        } else {
            ZIPFORMER_PARTIAL_INTERVAL_SAMPLES
        }
    }

    fn process_paraformer_stream(&mut self, samples: &[f32]) -> Result<LocalTranscriptResult> {
        let was_active = self.speech_active;
        if !was_active {
            self.update_preroll(samples);
        }
        let vad = self
            .vad
            .as_ref()
            .ok_or_else(|| anyhow!("Silero VAD is not initialized"))?;
        vad.accept_waveform(samples);
        let detected = vad.detected();
        let reached_endpoint = !self.take_ready_vad_segments().is_empty();
        let hypothesis = if !was_active && detected {
            let onset_audio = self.speech_preroll.clone();
            self.speech_preroll.clear();
            self.decode_paraformer_chunk(&onset_audio)?
        } else if was_active {
            self.decode_paraformer_chunk(samples)?
        } else {
            String::new()
        };
        self.speech_active = detected;

        if reached_endpoint {
            let result = LocalTranscriptResult {
                text: self.finish_paraformer_stream()?,
                is_final: true,
            };
            self.speech_preroll.clear();
            if !detected {
                self.update_preroll(samples);
            }
            return Ok(result);
        }

        if was_active && !detected {
            let next_stream = self
                .online_recognizer
                .as_ref()
                .ok_or_else(|| anyhow!("Paraformer online recognizer is not initialized"))?
                .create_stream();
            self.online_stream = Some(next_stream);
            self.online_last_text.clear();
            self.speech_preroll.clear();
            self.update_preroll(samples);
            return Ok(LocalTranscriptResult::default());
        }

        if detected && !hypothesis.is_empty() && hypothesis != self.online_last_text {
            self.online_last_text = hypothesis.clone();
            return Ok(LocalTranscriptResult {
                text: hypothesis,
                is_final: false,
            });
        }

        Ok(LocalTranscriptResult::default())
    }

    pub fn process(&mut self, audio_data: &[f32]) -> Result<LocalTranscriptResult> {
        let processed_data = if let Some(resampler) = &mut self.resampler {
            resampler.process(&[audio_data], None)?.remove(0)
        } else {
            audio_data.to_vec()
        };

        if self.model_id == PARAFORMER_STREAMING_MODEL_ID {
            return self.process_paraformer_stream(&processed_data);
        }

        if self.vad.is_some() {
            if !self.speech_active {
                self.update_preroll(&processed_data);
            }
            let was_active = self.speech_active;
            let vad = self.vad.as_ref().expect("checked above");
            vad.accept_waveform(&processed_data);
            let detected = vad.detected();

            if !was_active && detected {
                self.speech_active = true;
                self.speech_audio.clear();
                self.speech_audio.extend_from_slice(&self.speech_preroll);
                self.samples_at_last_partial = 0;
            } else if was_active {
                self.speech_audio.extend_from_slice(&processed_data);
            }

            let segments = self.take_ready_vad_segments();
            if !segments.is_empty() {
                let text = self.decode_final_segments(segments);
                self.speech_active = detected;
                self.speech_audio.clear();
                self.speech_preroll.clear();
                self.samples_at_last_partial = 0;
                self.update_preroll(&processed_data);
                if detected {
                    self.speech_audio.extend_from_slice(&self.speech_preroll);
                }
                return Ok(LocalTranscriptResult {
                    text,
                    is_final: true,
                });
            }

            if was_active && !detected {
                self.speech_active = false;
                self.speech_audio.clear();
                self.samples_at_last_partial = 0;
            }

            let partial_interval = self.partial_interval_samples();
            if self.speech_active
                && self
                    .speech_audio
                    .len()
                    .saturating_sub(self.samples_at_last_partial)
                    >= partial_interval
            {
                self.samples_at_last_partial = self.speech_audio.len();
                return Ok(LocalTranscriptResult {
                    text: self.decode_audio(&self.speech_audio),
                    is_final: false,
                });
            }

            return Ok(LocalTranscriptResult::default());
        }

        self.audio_buffer.extend_from_slice(&processed_data);
        if self.audio_buffer.len() < DEFAULT_CHUNK_SAMPLES {
            return Ok(LocalTranscriptResult::default());
        }

        let max_samples = MAX_BUFFER_SECONDS * TARGET_SAMPLE_RATE as usize;
        if self.audio_buffer.len() > max_samples {
            let drain_to = self.audio_buffer.len() - max_samples;
            self.audio_buffer.drain(..drain_to);
        }

        let audio_chunk = self.audio_buffer.drain(..).collect::<Vec<_>>();
        Ok(LocalTranscriptResult {
            text: self.finalize_text(self.decode_audio(&audio_chunk)),
            is_final: true,
        })
    }

    pub fn flush(&mut self) -> Result<LocalTranscriptResult> {
        if self.model_id == PARAFORMER_STREAMING_MODEL_ID {
            if let Some(vad) = &self.vad {
                vad.flush();
            }
            let text = self.finish_paraformer_stream()?;
            self.speech_active = false;
            if let Some(vad) = &self.vad {
                vad.clear();
                vad.reset();
            }
            return Ok(LocalTranscriptResult {
                text,
                is_final: true,
            });
        }

        if let Some(vad) = &self.vad {
            vad.flush();
            let segments = self.take_ready_vad_segments();
            let text = if segments.is_empty() && !self.speech_audio.is_empty() {
                self.finalize_text(self.decode_audio(&self.speech_audio))
            } else {
                self.decode_final_segments(segments)
            };
            self.speech_audio.clear();
            self.speech_preroll.clear();
            self.samples_at_last_partial = 0;
            self.speech_active = false;
            vad.clear();
            vad.reset();
            return Ok(LocalTranscriptResult {
                text,
                is_final: true,
            });
        }

        let audio_chunk = self.audio_buffer.drain(..).collect::<Vec<_>>();
        Ok(LocalTranscriptResult {
            text: self.finalize_text(self.decode_audio(&audio_chunk)),
            is_final: true,
        })
    }
}

pub struct AppState {
    pub model: Arc<Mutex<Option<SherpaOfflineModel>>>,
    pub vad_download: Mutex<()>,
    pub punctuation_download: Mutex<()>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            vad_download: Mutex::new(()),
            punctuation_download: Mutex::new(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        clean_transcript_text, ensure_terminal_punctuation, join_transcript_parts,
        normalize_sense_voice_language, LocalSpeechModelKind, SherpaOfflineModel,
        DEFAULT_LOCAL_SPEECH_MODEL_ID, PARAFORMER_STREAMING_MODEL_ID, SENSE_VOICE_MODEL_ID,
        ZIPFORMER_ZH_MODEL_ID,
    };
    use sherpa_onnx::Wave;
    use std::path::PathBuf;

    #[test]
    fn removes_unknown_tokens_from_transcripts() {
        assert_eq!(clean_transcript_text("你<unk>好"), "你好");
        assert_eq!(clean_transcript_text("hello <UNK> world"), "hello world");
    }

    #[test]
    fn validates_sense_voice_languages() {
        assert_eq!(normalize_sense_voice_language(None).unwrap(), "auto");
        assert_eq!(normalize_sense_voice_language(Some("zh-CN")).unwrap(), "zh");
        assert!(normalize_sense_voice_language(Some("fr")).is_err());
    }

    #[test]
    fn defaults_to_paraformer_streaming() {
        assert_eq!(DEFAULT_LOCAL_SPEECH_MODEL_ID, PARAFORMER_STREAMING_MODEL_ID);
        assert_eq!(
            LocalSpeechModelKind::from_id("").unwrap(),
            LocalSpeechModelKind::ParaformerStreaming
        );
    }

    #[test]
    fn joins_transcripts_without_spaces_between_chinese_sentences() {
        assert_eq!(
            join_transcript_parts(vec!["第一句。".to_string(), "第二句。".to_string()]),
            "第一句。第二句。"
        );
        assert_eq!(
            join_transcript_parts(vec!["hello".to_string(), "world".to_string()]),
            "hello world"
        );
    }

    #[test]
    fn adds_terminal_punctuation_when_the_model_omits_it() {
        assert_eq!(
            ensure_terminal_punctuation("今天天气不错".to_string()),
            "今天天气不错。"
        );
        assert_eq!(
            ensure_terminal_punctuation("looks good".to_string()),
            "looks good."
        );
        assert_eq!(
            ensure_terminal_punctuation("已经有标点！".to_string()),
            "已经有标点！"
        );
        assert_eq!(
            ensure_terminal_punctuation("“引用内容”".to_string()),
            "“引用内容。”"
        );
        assert_eq!(
            ensure_terminal_punctuation("下一句，".to_string()),
            "下一句，"
        );
    }

    #[test]
    #[ignore = "requires downloaded SenseVoice and Silero VAD models"]
    fn recognizes_speech_after_silero_vad_endpoint() {
        let model_dir = PathBuf::from(
            std::env::var("SENSE_VOICE_TEST_MODEL_DIR")
                .expect("SENSE_VOICE_TEST_MODEL_DIR is required"),
        );
        let vad_model = PathBuf::from(
            std::env::var("SILERO_VAD_TEST_MODEL").expect("SILERO_VAD_TEST_MODEL is required"),
        );
        let wave_path = model_dir.join("test_wavs").join("zh.wav");
        let wave = Wave::read(wave_path.to_str().expect("test wave path is not UTF-8"))
            .expect("failed to read SenseVoice test wave");
        let mut model = SherpaOfflineModel::new(
            &model_dir,
            SENSE_VOICE_MODEL_ID,
            Some("zh"),
            Some(&vad_model),
            None,
            wave.sample_rate() as u32,
        )
        .expect("failed to create SenseVoice with Silero VAD");

        let mut transcripts = Vec::new();
        let mut saw_partial = false;
        let mut saw_final = false;
        for chunk in wave.samples().chunks(1024) {
            let result = model.process(chunk).expect("failed to process speech");
            if !result.text.is_empty() {
                saw_partial |= !result.is_final;
                saw_final |= result.is_final;
                transcripts.push(result.text);
            }
        }
        for chunk in vec![0.0_f32; 16_000].chunks(1024) {
            let result = model.process(chunk).expect("failed to process silence");
            if !result.text.is_empty() {
                saw_partial |= !result.is_final;
                saw_final |= result.is_final;
                transcripts.push(result.text);
            }
        }
        let tail = model.flush().expect("failed to flush VAD");
        if !tail.text.is_empty() {
            saw_final |= tail.is_final;
            transcripts.push(tail.text);
        }

        eprintln!("SenseVoice VAD output: {}", transcripts.join(" | "));
        assert!(!transcripts.join(" ").trim().is_empty());
        assert!(saw_partial, "SenseVoice did not emit a partial result");
        assert!(saw_final, "SenseVoice did not emit a final result");
    }

    #[test]
    #[ignore = "requires downloaded Paraformer, Silero VAD, and punctuation models"]
    fn paraformer_streams_then_restores_punctuation() {
        let model_dir = PathBuf::from(
            std::env::var("PARAFORMER_TEST_MODEL_DIR")
                .expect("PARAFORMER_TEST_MODEL_DIR is required"),
        );
        let vad_model = PathBuf::from(
            std::env::var("SILERO_VAD_TEST_MODEL").expect("SILERO_VAD_TEST_MODEL is required"),
        );
        let punctuation_model = PathBuf::from(
            std::env::var("PUNCTUATION_TEST_MODEL").expect("PUNCTUATION_TEST_MODEL is required"),
        );
        let wave_path = model_dir.join("test_wavs").join("0.wav");
        let wave = Wave::read(wave_path.to_str().expect("test wave path is not UTF-8"))
            .expect("failed to read Paraformer test wave");
        let mut model = SherpaOfflineModel::new(
            &model_dir,
            PARAFORMER_STREAMING_MODEL_ID,
            None,
            Some(&vad_model),
            Some(&punctuation_model),
            wave.sample_rate() as u32,
        )
        .expect("failed to create Paraformer with VAD and punctuation");

        let mut partials = Vec::new();
        let mut finals = Vec::new();
        for chunk in wave
            .samples()
            .iter()
            .copied()
            .chain(std::iter::repeat(0.0).take(16_000))
            .collect::<Vec<_>>()
            .chunks(1024)
        {
            let result = model.process(chunk).expect("failed to process audio");
            if result.text.is_empty() {
                continue;
            }
            if result.is_final {
                finals.push(result.text);
            } else {
                partials.push(result.text);
            }
        }
        let tail = model.flush().expect("failed to flush Paraformer");
        if !tail.text.is_empty() {
            finals.push(tail.text);
        }

        let final_text = finals.join(" ");
        eprintln!("Paraformer partials: {}", partials.join(" | "));
        eprintln!("Paraformer final: {final_text}");
        assert!(!partials.is_empty(), "Paraformer emitted no partial result");
        assert!(!final_text.is_empty(), "Paraformer emitted no final result");
        assert!(
            final_text
                .chars()
                .any(|character| "，。！？；：,.!?;:".contains(character)),
            "Paraformer final result did not contain punctuation: {final_text}"
        );
    }

    #[test]
    #[ignore = "requires downloaded Zipformer, Silero VAD, and punctuation models"]
    fn zipformer_streams_then_restores_punctuation() {
        let model_dir = PathBuf::from(
            std::env::var("ZIPFORMER_TEST_MODEL_DIR")
                .expect("ZIPFORMER_TEST_MODEL_DIR is required"),
        );
        let vad_model = PathBuf::from(
            std::env::var("SILERO_VAD_TEST_MODEL").expect("SILERO_VAD_TEST_MODEL is required"),
        );
        let punctuation_model = PathBuf::from(
            std::env::var("PUNCTUATION_TEST_MODEL").expect("PUNCTUATION_TEST_MODEL is required"),
        );
        let wave_path = model_dir.join("test_wavs").join("0.wav");
        let wave = Wave::read(wave_path.to_str().expect("test wave path is not UTF-8"))
            .expect("failed to read Zipformer test wave");
        let mut model = SherpaOfflineModel::new(
            &model_dir,
            ZIPFORMER_ZH_MODEL_ID,
            None,
            Some(&vad_model),
            Some(&punctuation_model),
            wave.sample_rate() as u32,
        )
        .expect("failed to create Zipformer with VAD and punctuation");

        let mut partials = Vec::new();
        let mut finals = Vec::new();
        for chunk in wave
            .samples()
            .iter()
            .copied()
            .chain(std::iter::repeat(0.0).take(16_000))
            .collect::<Vec<_>>()
            .chunks(1024)
        {
            let result = model.process(chunk).expect("failed to process audio");
            if result.text.is_empty() {
                continue;
            }
            if result.is_final {
                finals.push(result.text);
            } else {
                partials.push(result.text);
            }
        }
        let tail = model.flush().expect("failed to flush Zipformer VAD");
        if !tail.text.is_empty() {
            finals.push(tail.text);
        }

        let final_text = finals.join(" ");
        eprintln!("Zipformer partials: {}", partials.join(" | "));
        eprintln!("Zipformer final: {final_text}");
        assert!(
            !partials.is_empty(),
            "Zipformer did not emit a partial result"
        );
        assert!(
            !final_text.is_empty(),
            "Zipformer did not emit a final result"
        );
        assert!(
            final_text
                .chars()
                .any(|character| "，。！？；：,.!?;:".contains(character)),
            "Zipformer final result did not contain punctuation: {final_text}"
        );
    }
}
