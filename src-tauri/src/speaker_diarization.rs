use serde::{Deserialize, Serialize};
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};
use tokio::process::Command;
use tokio::sync::Mutex as AsyncMutex;

const MODEL_ROOT: &str = "speaker-diarization";
const SEGMENTATION_DIR: &str = "sherpa-onnx-pyannote-segmentation-3-0";
const SEGMENTATION_FILE: &str = "model.onnx";
const SEGMENTATION_ARCHIVE_URLS: [&str; 3] = [
    "https://gh-proxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    "https://ghfast.top/https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
];
const EMBEDDING_FILE: &str = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";
const EMBEDDING_URLS: [&str; 3] = [
    "https://gh-proxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "https://ghfast.top/https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
];
const REALTIME_WINDOW_SAMPLES: usize = 48_000;
const REALTIME_INTERVAL_SAMPLES: usize = 24_000;
const REALTIME_MIN_RMS: f32 = 0.006;
const REALTIME_MATCH_THRESHOLD: f32 = 0.48;
const REALTIME_NEW_SPEAKER_THRESHOLD: f32 = 0.65;
const REALTIME_MAX_SPEAKERS: usize = 6;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker: u32,
}

#[derive(Clone, Debug, Default)]
pub struct SpeakerDiarizationOutput {
    pub speaker_count: u32,
    pub segments: Vec<SpeakerSegment>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RealtimeSpeakerEvent {
    speaker: u32,
    speaker_count: u32,
    confidence: f32,
    analyzed_end_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RealtimeSpeakerStatusEvent {
    status: String,
    error: Option<String>,
}

#[derive(Default)]
struct SpeakerCentroid {
    embedding: Vec<f32>,
    observations: u32,
}

#[derive(Default)]
struct RealtimeSpeakerSession {
    generation: u64,
    active: bool,
    model_ready: bool,
    samples: Vec<f32>,
    total_samples: u64,
    samples_since_analysis: usize,
    analysis_in_flight: bool,
    centroids: Vec<SpeakerCentroid>,
    pending_unknown: Option<Vec<f32>>,
}

struct RealtimeClassification {
    speaker: u32,
    confidence: f32,
}

#[derive(Clone, Default)]
pub struct SpeakerDiarizationState {
    download_lock: Arc<AsyncMutex<()>>,
    engine: Arc<Mutex<Option<OfflineSpeakerDiarization>>>,
    embedding_extractor: Arc<Mutex<Option<SpeakerEmbeddingExtractor>>>,
    realtime: Arc<Mutex<RealtimeSpeakerSession>>,
}

impl SpeakerDiarizationState {
    pub fn new() -> Self {
        Self::default()
    }
}

fn model_root(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map(|path| path.join("models").join(MODEL_ROOT))
        .map_err(|error| error.to_string())
}

fn segmentation_model_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(model_root(app)?
        .join(SEGMENTATION_DIR)
        .join(SEGMENTATION_FILE))
}

fn embedding_model_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(model_root(app)?.join(EMBEDDING_FILE))
}

fn file_is_ready(path: &Path, minimum_size: u64) -> bool {
    path.is_file()
        && path
            .metadata()
            .map(|metadata| metadata.len() >= minimum_size)
            .unwrap_or(false)
}

async fn download_bytes(
    client: &reqwest::Client,
    urls: &[&str],
    minimum_size: usize,
    label: &str,
) -> Result<Vec<u8>, String> {
    let mut last_error = String::new();
    for url in urls {
        match client.get(*url).send().await {
            Ok(response) => match response.error_for_status() {
                Ok(response) => match response.bytes().await {
                    Ok(bytes) if bytes.len() >= minimum_size => return Ok(bytes.to_vec()),
                    Ok(bytes) => {
                        last_error = format!("下载文件不完整（{} 字节）", bytes.len());
                    }
                    Err(error) => last_error = error.to_string(),
                },
                Err(error) => last_error = error.to_string(),
            },
            Err(error) => last_error = error.to_string(),
        }
    }
    Err(format!("下载{label}失败：{last_error}"))
}

async fn ensure_models(app: &AppHandle) -> Result<(PathBuf, PathBuf), String> {
    let segmentation_path = segmentation_model_path(app)?;
    let embedding_path = embedding_model_path(app)?;
    if file_is_ready(&segmentation_path, 5 * 1024 * 1024)
        && file_is_ready(&embedding_path, 30 * 1024 * 1024)
    {
        return Ok((segmentation_path, embedding_path));
    }

    let state = app.state::<SpeakerDiarizationState>();
    let _guard = state.download_lock.lock().await;
    if file_is_ready(&segmentation_path, 5 * 1024 * 1024)
        && file_is_ready(&embedding_path, 30 * 1024 * 1024)
    {
        return Ok((segmentation_path, embedding_path));
    }

    let root = model_root(app)?;
    tokio::fs::create_dir_all(&root)
        .await
        .map_err(|error| format!("创建说话人分离模型目录失败：{error}"))?;
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|error| format!("创建模型下载客户端失败：{error}"))?;

    if !file_is_ready(&segmentation_path, 5 * 1024 * 1024) {
        let bytes = download_bytes(
            &client,
            &SEGMENTATION_ARCHIVE_URLS,
            5 * 1024 * 1024,
            "说话人分段模型",
        )
        .await?;
        let archive_path = root.join(format!("{SEGMENTATION_DIR}.tar.bz2.part"));
        tokio::fs::write(&archive_path, bytes)
            .await
            .map_err(|error| format!("保存说话人分段模型失败：{error}"))?;
        let output = Command::new("tar")
            .arg("-xjf")
            .arg(&archive_path)
            .arg("-C")
            .arg(&root)
            .output()
            .await
            .map_err(|error| format!("无法启动说话人分段模型解压程序：{error}"))?;
        let _ = tokio::fs::remove_file(&archive_path).await;
        if !output.status.success() {
            return Err(format!(
                "解压说话人分段模型失败：{}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        if !file_is_ready(&segmentation_path, 5 * 1024 * 1024) {
            return Err("解压后的说话人分段模型文件不完整".to_string());
        }
    }

    if !file_is_ready(&embedding_path, 30 * 1024 * 1024) {
        let bytes =
            download_bytes(&client, &EMBEDDING_URLS, 30 * 1024 * 1024, "音色特征模型").await?;
        let partial_path = root.join(format!("{EMBEDDING_FILE}.part"));
        tokio::fs::write(&partial_path, bytes)
            .await
            .map_err(|error| format!("保存音色特征模型失败：{error}"))?;
        if embedding_path.exists() {
            tokio::fs::remove_file(&embedding_path)
                .await
                .map_err(|error| format!("替换音色特征模型失败：{error}"))?;
        }
        tokio::fs::rename(&partial_path, &embedding_path)
            .await
            .map_err(|error| format!("完成音色特征模型缓存失败：{error}"))?;
    }

    Ok((segmentation_path, embedding_path))
}

async fn ensure_embedding_model(app: &AppHandle) -> Result<PathBuf, String> {
    let embedding_path = embedding_model_path(app)?;
    if file_is_ready(&embedding_path, 30 * 1024 * 1024) {
        return Ok(embedding_path);
    }

    let state = app.state::<SpeakerDiarizationState>();
    let _guard = state.download_lock.lock().await;
    if file_is_ready(&embedding_path, 30 * 1024 * 1024) {
        return Ok(embedding_path);
    }

    let root = model_root(app)?;
    tokio::fs::create_dir_all(&root)
        .await
        .map_err(|error| format!("创建说话人分离模型目录失败：{error}"))?;
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|error| format!("创建模型下载客户端失败：{error}"))?;
    let bytes = download_bytes(&client, &EMBEDDING_URLS, 30 * 1024 * 1024, "音色特征模型").await?;
    let partial_path = root.join(format!("{EMBEDDING_FILE}.part"));
    tokio::fs::write(&partial_path, bytes)
        .await
        .map_err(|error| format!("保存音色特征模型失败：{error}"))?;
    if embedding_path.exists() {
        tokio::fs::remove_file(&embedding_path)
            .await
            .map_err(|error| format!("替换音色特征模型失败：{error}"))?;
    }
    tokio::fs::rename(&partial_path, &embedding_path)
        .await
        .map_err(|error| format!("完成音色特征模型缓存失败：{error}"))?;
    Ok(embedding_path)
}

fn read_recording(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader =
        hound::WavReader::open(path).map_err(|error| format!("读取录音文件失败：{error}"))?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000
        || spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        return Err(format!(
            "说话人分离需要 16 kHz 单声道 PCM 录音，当前为 {} Hz / {} 声道 / {} bit",
            spec.sample_rate, spec.channels, spec.bits_per_sample
        ));
    }

    reader
        .samples::<i16>()
        .map(|sample| {
            sample
                .map(|value| value as f32 / i16::MAX as f32)
                .map_err(|error| format!("解析录音采样失败：{error}"))
        })
        .collect()
}

fn normalize_segments(
    segments: Vec<sherpa_onnx::OfflineSpeakerDiarizationSegment>,
) -> SpeakerDiarizationOutput {
    let mut labels = HashMap::<i32, u32>::new();
    let mut next_label = 0_u32;
    let mut normalized = Vec::<SpeakerSegment>::new();

    for segment in segments {
        let start_ms = (segment.start.max(0.0) * 1_000.0).round() as u64;
        let end_ms = (segment.end.max(segment.start) * 1_000.0).round() as u64;
        if end_ms <= start_ms {
            continue;
        }
        let speaker = *labels.entry(segment.speaker).or_insert_with(|| {
            let label = next_label;
            next_label += 1;
            label
        });

        if let Some(previous) = normalized.last_mut() {
            if previous.speaker == speaker && start_ms <= previous.end_ms.saturating_add(200) {
                previous.end_ms = previous.end_ms.max(end_ms);
                continue;
            }
        }
        normalized.push(SpeakerSegment {
            start_ms,
            end_ms,
            speaker,
        });
    }

    let speaker_count = normalized
        .iter()
        .map(|segment| segment.speaker)
        .collect::<HashSet<_>>()
        .len() as u32;
    SpeakerDiarizationOutput {
        speaker_count,
        segments: normalized,
    }
}

fn normalize_embedding(mut embedding: Vec<f32>) -> Option<Vec<f32>> {
    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return None;
    }
    for value in &mut embedding {
        *value /= norm;
    }
    Some(embedding)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return -1.0;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn update_centroid(centroid: &mut SpeakerCentroid, embedding: &[f32]) {
    let weight = if centroid.observations < 5 {
        1.0 / (centroid.observations + 1) as f32
    } else {
        0.12
    };
    for (current, incoming) in centroid.embedding.iter_mut().zip(embedding) {
        *current = *current * (1.0 - weight) + *incoming * weight;
    }
    if let Some(normalized) = normalize_embedding(std::mem::take(&mut centroid.embedding)) {
        centroid.embedding = normalized;
    }
    centroid.observations += 1;
}

fn classify_realtime_embedding(
    session: &mut RealtimeSpeakerSession,
    embedding: Vec<f32>,
) -> Option<RealtimeClassification> {
    let embedding = normalize_embedding(embedding)?;
    if session.centroids.is_empty() {
        session.centroids.push(SpeakerCentroid {
            embedding,
            observations: 1,
        });
        return Some(RealtimeClassification {
            speaker: 0,
            confidence: 1.0,
        });
    }

    let (best_speaker, best_score) = session
        .centroids
        .iter()
        .enumerate()
        .map(|(speaker, centroid)| (speaker, cosine_similarity(&centroid.embedding, &embedding)))
        .max_by(|left, right| left.1.total_cmp(&right.1))?;

    if best_score >= REALTIME_MATCH_THRESHOLD || session.centroids.len() >= REALTIME_MAX_SPEAKERS {
        session.pending_unknown = None;
        if best_score >= REALTIME_MATCH_THRESHOLD {
            update_centroid(&mut session.centroids[best_speaker], &embedding);
        }
        return Some(RealtimeClassification {
            speaker: best_speaker as u32,
            confidence: best_score.clamp(0.0, 1.0),
        });
    }

    let Some(pending) = session.pending_unknown.take() else {
        session.pending_unknown = Some(embedding);
        return None;
    };
    let pending_score = cosine_similarity(&pending, &embedding);
    if pending_score < REALTIME_NEW_SPEAKER_THRESHOLD {
        session.pending_unknown = Some(embedding);
        return None;
    }

    let combined = pending
        .iter()
        .zip(&embedding)
        .map(|(left, right)| (left + right) * 0.5)
        .collect::<Vec<_>>();
    session.centroids.push(SpeakerCentroid {
        embedding: normalize_embedding(combined)?,
        observations: 2,
    });
    Some(RealtimeClassification {
        speaker: (session.centroids.len() - 1) as u32,
        confidence: pending_score.clamp(0.0, 1.0),
    })
}

fn emit_realtime_status(app: &AppHandle, status: &str, error: Option<String>) {
    let _ = app.emit(
        "realtime-speaker-status",
        RealtimeSpeakerStatusEvent {
            status: status.to_string(),
            error,
        },
    );
}

pub fn start_realtime(app: &AppHandle, enabled: bool) {
    let state = app.state::<SpeakerDiarizationState>().inner().clone();
    let generation = {
        let Ok(mut session) = state.realtime.lock() else {
            emit_realtime_status(app, "failed", Some("实时音色分析状态不可用".to_string()));
            return;
        };
        let generation = session.generation.wrapping_add(1);
        *session = RealtimeSpeakerSession {
            generation,
            active: enabled,
            ..Default::default()
        };
        generation
    };

    if !enabled {
        emit_realtime_status(app, "idle", None);
        return;
    }
    emit_realtime_status(app, "loading", None);

    let app_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        let result = async {
            let embedding_path = ensure_embedding_model(&app_for_task).await?;
            let state_for_model = state.clone();
            tauri::async_runtime::spawn_blocking(move || {
                let mut extractor = state_for_model
                    .embedding_extractor
                    .lock()
                    .map_err(|_| "实时音色模型状态不可用".to_string())?;
                if extractor.is_none() {
                    *extractor = Some(
                        SpeakerEmbeddingExtractor::create(&SpeakerEmbeddingExtractorConfig {
                            model: Some(embedding_path.to_string_lossy().into_owned()),
                            num_threads: 1,
                            ..Default::default()
                        })
                        .ok_or_else(|| "无法初始化实时音色特征模型".to_string())?,
                    );
                }
                Ok::<(), String>(())
            })
            .await
            .map_err(|error| format!("实时音色模型加载任务异常结束：{error}"))??;
            Ok::<(), String>(())
        }
        .await;

        let current = state
            .realtime
            .lock()
            .map(|mut session| {
                if session.active && session.generation == generation {
                    session.model_ready = result.is_ok();
                    if result.is_err() {
                        session.active = false;
                    }
                    true
                } else {
                    false
                }
            })
            .unwrap_or(false);
        if !current {
            return;
        }
        match result {
            Ok(()) => emit_realtime_status(&app_for_task, "ready", None),
            Err(error) => emit_realtime_status(&app_for_task, "failed", Some(error)),
        }
    });
}

pub fn stop_realtime(app: &AppHandle) {
    if let Ok(mut session) = app.state::<SpeakerDiarizationState>().realtime.lock() {
        let generation = session.generation.wrapping_add(1);
        *session = RealtimeSpeakerSession {
            generation,
            ..Default::default()
        };
    }
    emit_realtime_status(app, "idle", None);
}

pub fn append_realtime_samples(app: &AppHandle, samples: &[f32]) {
    if samples.is_empty() {
        return;
    }
    let state = app.state::<SpeakerDiarizationState>().inner().clone();
    let job = {
        let Ok(mut session) = state.realtime.lock() else {
            return;
        };
        if !session.active {
            return;
        }

        session.total_samples = session.total_samples.saturating_add(samples.len() as u64);
        session.samples_since_analysis =
            session.samples_since_analysis.saturating_add(samples.len());
        session.samples.extend_from_slice(samples);
        if session.samples.len() > REALTIME_WINDOW_SAMPLES {
            let overflow = session.samples.len() - REALTIME_WINDOW_SAMPLES;
            session.samples.drain(..overflow);
        }

        if !session.model_ready
            || session.analysis_in_flight
            || session.samples.len() < REALTIME_WINDOW_SAMPLES
            || session.samples_since_analysis < REALTIME_INTERVAL_SAMPLES
        {
            None
        } else {
            session.samples_since_analysis = 0;
            let mean_square = session
                .samples
                .iter()
                .map(|sample| sample * sample)
                .sum::<f32>()
                / session.samples.len() as f32;
            if mean_square.sqrt() < REALTIME_MIN_RMS {
                None
            } else {
                session.analysis_in_flight = true;
                Some((
                    session.generation,
                    session.samples.clone(),
                    session.total_samples.saturating_mul(1_000) / 16_000,
                ))
            }
        }
    };

    let Some((generation, window, analyzed_end_ms)) = job else {
        return;
    };
    let app_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        let state_for_model = state.clone();
        let result = tauri::async_runtime::spawn_blocking(move || {
            let extractor = state_for_model
                .embedding_extractor
                .lock()
                .map_err(|_| "实时音色模型状态不可用".to_string())?;
            let extractor = extractor
                .as_ref()
                .ok_or_else(|| "实时音色模型尚未加载".to_string())?;
            let stream = extractor
                .create_stream()
                .ok_or_else(|| "无法创建实时音色分析流".to_string())?;
            stream.accept_waveform(16_000, &window);
            if !extractor.is_ready(&stream) {
                return Err("当前语音片段过短，无法提取音色".to_string());
            }
            extractor
                .compute(&stream)
                .ok_or_else(|| "实时音色特征提取失败".to_string())
        })
        .await
        .map_err(|error| format!("实时音色分析任务异常结束：{error}"))
        .and_then(|result| result);

        let update = state.realtime.lock().ok().and_then(|mut session| {
            if !session.active || session.generation != generation {
                return None;
            }
            session.analysis_in_flight = false;
            match result {
                Ok(embedding) => {
                    classify_realtime_embedding(&mut session, embedding).map(|classification| {
                        RealtimeSpeakerEvent {
                            speaker: classification.speaker,
                            speaker_count: session.centroids.len() as u32,
                            confidence: classification.confidence,
                            analyzed_end_ms,
                        }
                    })
                }
                Err(error) => {
                    session.active = false;
                    emit_realtime_status(&app_for_task, "failed", Some(error));
                    None
                }
            }
        });
        if let Some(event) = update {
            let _ = app_for_task.emit("realtime-speaker", event);
        }
    });
}

pub async fn diarize_recording_audio(
    app: &AppHandle,
    audio_path: PathBuf,
) -> Result<SpeakerDiarizationOutput, String> {
    let (segmentation_path, embedding_path) = ensure_models(app).await?;
    let state = app.state::<SpeakerDiarizationState>().inner().clone();

    tauri::async_runtime::spawn_blocking(move || {
        let samples = read_recording(&audio_path)?;
        if samples.is_empty() {
            return Ok(SpeakerDiarizationOutput::default());
        }

        let mut engine_guard = state
            .engine
            .lock()
            .map_err(|_| "说话人分离引擎状态不可用".to_string())?;
        if engine_guard.is_none() {
            let config = OfflineSpeakerDiarizationConfig {
                segmentation: OfflineSpeakerSegmentationModelConfig {
                    pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                        model: Some(segmentation_path.to_string_lossy().into_owned()),
                    },
                    num_threads: 2,
                    ..Default::default()
                },
                embedding: SpeakerEmbeddingExtractorConfig {
                    model: Some(embedding_path.to_string_lossy().into_owned()),
                    num_threads: 2,
                    ..Default::default()
                },
                clustering: FastClusteringConfig {
                    num_clusters: -1,
                    threshold: 0.5,
                },
                ..Default::default()
            };
            *engine_guard = Some(
                OfflineSpeakerDiarization::create(&config)
                    .ok_or_else(|| "无法初始化本地说话人分离引擎".to_string())?,
            );
        }

        let engine = engine_guard.as_ref().expect("initialized above");
        if engine.sample_rate() != 16_000 {
            return Err(format!(
                "说话人分离模型采样率不匹配：{} Hz",
                engine.sample_rate()
            ));
        }
        let result = engine
            .process(&samples)
            .ok_or_else(|| "本地说话人分离失败".to_string())?;
        Ok(normalize_segments(result.sort_by_start_time()))
    })
    .await
    .map_err(|error| format!("说话人分离任务异常结束：{error}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_labels_and_merges_adjacent_segments() {
        let output = normalize_segments(vec![
            sherpa_onnx::OfflineSpeakerDiarizationSegment {
                start: 0.0,
                end: 1.0,
                speaker: 8,
            },
            sherpa_onnx::OfflineSpeakerDiarizationSegment {
                start: 1.1,
                end: 2.0,
                speaker: 8,
            },
            sherpa_onnx::OfflineSpeakerDiarizationSegment {
                start: 2.2,
                end: 3.0,
                speaker: 3,
            },
        ]);

        assert_eq!(output.speaker_count, 2);
        assert_eq!(output.segments.len(), 2);
        assert_eq!(output.segments[0].speaker, 0);
        assert_eq!(output.segments[0].end_ms, 2_000);
        assert_eq!(output.segments[1].speaker, 1);
    }

    #[test]
    fn realtime_clustering_requires_a_repeated_new_voice() {
        let mut session = RealtimeSpeakerSession::default();
        let first = classify_realtime_embedding(&mut session, vec![1.0, 0.0, 0.0]).unwrap();
        let same = classify_realtime_embedding(&mut session, vec![0.98, 0.08, 0.0]).unwrap();

        assert_eq!(first.speaker, 0);
        assert_eq!(same.speaker, 0);
        assert_eq!(session.centroids.len(), 1);
        assert!(classify_realtime_embedding(&mut session, vec![0.0, 1.0, 0.0]).is_none());

        let second = classify_realtime_embedding(&mut session, vec![0.04, 0.99, 0.0]).unwrap();
        assert_eq!(second.speaker, 1);
        assert_eq!(session.centroids.len(), 2);
    }

    #[test]
    #[ignore = "requires a downloaded embedding model and an audio sample"]
    fn classifies_official_sample_in_realtime_windows() {
        let embedding_model = std::env::var("SHERPA_DIARIZATION_EMBEDDING_MODEL").unwrap();
        let audio_path = std::env::var("SHERPA_DIARIZATION_TEST_WAV").unwrap();
        let samples = read_recording(Path::new(&audio_path)).unwrap();
        let extractor = SpeakerEmbeddingExtractor::create(&SpeakerEmbeddingExtractorConfig {
            model: Some(embedding_model),
            ..Default::default()
        })
        .unwrap();
        let mut session = RealtimeSpeakerSession::default();
        let mut assignments = Vec::new();

        for end in (REALTIME_WINDOW_SAMPLES..=samples.len()).step_by(REALTIME_INTERVAL_SAMPLES) {
            let window = &samples[end - REALTIME_WINDOW_SAMPLES..end];
            let rms = (window.iter().map(|sample| sample * sample).sum::<f32>()
                / window.len() as f32)
                .sqrt();
            if rms < REALTIME_MIN_RMS {
                continue;
            }
            let stream = extractor.create_stream().unwrap();
            stream.accept_waveform(16_000, window);
            if !extractor.is_ready(&stream) {
                continue;
            }
            let embedding = extractor.compute(&stream).unwrap();
            let normalized = normalize_embedding(embedding.clone()).unwrap();
            let scores = session
                .centroids
                .iter()
                .map(|centroid| cosine_similarity(&centroid.embedding, &normalized))
                .collect::<Vec<_>>();
            if let Some(classification) = classify_realtime_embedding(&mut session, embedding) {
                eprintln!(
                    "window_end={:.1}s scores={scores:?} -> {} ({:.3})",
                    end as f32 / 16_000.0,
                    classification.speaker,
                    classification.confidence
                );
                assignments.push(classification.speaker);
            } else {
                eprintln!(
                    "window_end={:.1}s scores={scores:?} -> pending",
                    end as f32 / 16_000.0
                );
            }
        }

        eprintln!(
            "Realtime window assignments: {assignments:?}; speakers: {}",
            session.centroids.len()
        );
        assert!(session.centroids.len() >= 2);
        assert!(!assignments.is_empty());
    }

    #[test]
    #[ignore = "requires downloaded diarization models and an audio sample"]
    fn diarizes_official_sample() {
        let segmentation_model = std::env::var("SHERPA_DIARIZATION_SEGMENTATION_MODEL").unwrap();
        let embedding_model = std::env::var("SHERPA_DIARIZATION_EMBEDDING_MODEL").unwrap();
        let audio_path = std::env::var("SHERPA_DIARIZATION_TEST_WAV").unwrap();
        let samples = read_recording(Path::new(&audio_path)).unwrap();
        let config = OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: Some(segmentation_model),
                },
                ..Default::default()
            },
            embedding: SpeakerEmbeddingExtractorConfig {
                model: Some(embedding_model),
                ..Default::default()
            },
            clustering: FastClusteringConfig {
                num_clusters: -1,
                threshold: 0.5,
            },
            ..Default::default()
        };
        let engine = OfflineSpeakerDiarization::create(&config).unwrap();
        let result = engine.process(&samples).unwrap();
        let output = normalize_segments(result.sort_by_start_time());

        eprintln!(
            "Offline diarization speakers: {}; segments: {:?}",
            output.speaker_count, output.segments
        );
        assert!(output.speaker_count >= 2);
        assert!(!output.segments.is_empty());
    }
}
