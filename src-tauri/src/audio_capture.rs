use crate::inference::AppState;
use crate::recording_store::RecordingStoreState;
use crate::speaker_diarization;
use serde::Serialize;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::Mutex;

#[derive(Clone, Default)]
pub struct AudioCaptureState {
    process_capture: Arc<Mutex<Option<CaptureHandle>>>,
}

impl AudioCaptureState {
    pub fn new() -> Self {
        Self::default()
    }
}

struct CaptureHandle {
    stop: Arc<AtomicBool>,
    worker: std::thread::JoinHandle<()>,
}

impl CaptureHandle {
    fn stop(self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = self.worker.join();
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioTranscriptEvent {
    pub source: String,
    pub text: String,
    pub is_final: bool,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioChunkEvent {
    pub source: String,
    pub samples: Vec<f32>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioCaptureStatsEvent {
    pub source: String,
    pub chunks: u64,
    pub samples: u64,
    pub rms: f32,
    pub peak: f32,
    pub silent: bool,
}

#[tauri::command]
pub async fn start_process_audio_capture(
    app: AppHandle,
    audio_state: tauri::State<'_, AudioCaptureState>,
    pid: u32,
    persist_audio: Option<bool>,
) -> Result<(), String> {
    let mut guard = audio_state.process_capture.lock().await;
    if guard.is_some() {
        return Err("Process audio capture is already running".to_string());
    }

    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = stop.clone();
    let app_for_worker = app.clone();

    let worker = std::thread::Builder::new()
        .name("process-audio-capture".to_string())
        .spawn(move || {
            let result = process_audio_capture_worker(
                app_for_worker.clone(),
                pid,
                worker_stop,
                persist_audio.unwrap_or(true),
            );
            if let Err(error) = result {
                let _ = app_for_worker.emit("audio-transcription-error", error);
            }
        })
        .map_err(|e| e.to_string())?;

    *guard = Some(CaptureHandle { stop, worker });
    Ok(())
}

#[tauri::command]
pub async fn start_process_audio_transcription(
    app: AppHandle,
    audio_state: tauri::State<'_, AudioCaptureState>,
    pid: u32,
) -> Result<(), String> {
    let mut guard = audio_state.process_capture.lock().await;
    if guard.is_some() {
        return Err("Process audio transcription is already running".to_string());
    }

    let inference_state = app.state::<AppState>();
    if inference_state.model.lock().await.is_none() {
        return Err(
            "Local Sherpa-ONNX model is not loaded. Download/load the local model first."
                .to_string(),
        );
    }

    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = stop.clone();
    let app_for_worker = app.clone();

    let worker = std::thread::Builder::new()
        .name("process-audio-transcription".to_string())
        .spawn(move || {
            let result = process_audio_worker(app_for_worker.clone(), pid, worker_stop);
            if let Err(error) = result {
                let _ = app_for_worker.emit("audio-transcription-error", error);
            }
        })
        .map_err(|e| e.to_string())?;

    *guard = Some(CaptureHandle { stop, worker });
    Ok(())
}

#[tauri::command]
pub async fn stop_process_audio_transcription(
    audio_state: tauri::State<'_, AudioCaptureState>,
) -> Result<(), String> {
    let mut guard = audio_state.process_capture.lock().await;
    if let Some(handle) = guard.take() {
        tauri::async_runtime::spawn_blocking(move || handle.stop())
            .await
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub fn is_process_audio_capture_supported() -> bool {
    cfg!(windows)
}

fn process_audio_capture_worker(
    app: AppHandle,
    pid: u32,
    stop: Arc<AtomicBool>,
    persist_audio: bool,
) -> Result<(), String> {
    #[cfg(windows)]
    {
        windows_process_loopback::run_raw(app, pid, stop, persist_audio)
            .map_err(describe_windows_loopback_error)
    }

    #[cfg(not(windows))]
    {
        let _ = app;
        let _ = pid;
        let _ = stop;
        let _ = persist_audio;
        Err("Process audio capture is currently implemented only on Windows.".to_string())
    }
}

fn process_audio_worker(app: AppHandle, pid: u32, stop: Arc<AtomicBool>) -> Result<(), String> {
    #[cfg(windows)]
    {
        windows_process_loopback::run(app, pid, stop).map_err(describe_windows_loopback_error)
    }

    #[cfg(not(windows))]
    {
        let _ = app;
        let _ = pid;
        let _ = stop;
        Err("Process audio capture is currently implemented only on Windows.".to_string())
    }
}

#[cfg(windows)]
fn describe_windows_loopback_error(error: String) -> String {
    if error.contains("0x80004001") {
        return format!(
            "Windows process audio loopback is not implemented in this audio environment. \
             This feature requires Windows Application Loopback support and an active audio output device. Raw error: {error}"
        );
    }

    error
}

fn process_chunk(app: &AppHandle, chunk: &[f32]) -> Result<(), String> {
    app.state::<RecordingStoreState>().append_samples(chunk)?;
    speaker_diarization::append_realtime_samples(app, chunk);
    let state = app.state::<AppState>();
    let model = state.model.clone();
    let samples = chunk.to_vec();
    let app_for_event = app.clone();

    tauri::async_runtime::block_on(async move {
        let mut model_guard = model.lock().await;
        let Some(model) = model_guard.as_mut() else {
            return Err("Local Sherpa-ONNX model is not loaded.".to_string());
        };

        let result = model.process(&samples).map_err(|e| e.to_string())?;
        if !result.text.trim().is_empty() {
            app_for_event
                .emit(
                    "audio-transcript",
                    AudioTranscriptEvent {
                        source: "process".to_string(),
                        text: result.text,
                        is_final: result.is_final,
                    },
                )
                .map_err(|e| e.to_string())?;
        }

        Ok(())
    })
}

fn emit_audio_chunk(app: &AppHandle, chunk: &[f32], persist_audio: bool) -> Result<(), String> {
    if persist_audio {
        app.state::<RecordingStoreState>().append_samples(chunk)?;
        speaker_diarization::append_realtime_samples(app, chunk);
    }
    app.emit(
        "audio-chunk",
        AudioChunkEvent {
            source: "process".to_string(),
            samples: chunk.to_vec(),
        },
    )
    .map_err(|e| e.to_string())
}

fn emit_audio_capture_stats(
    app: &AppHandle,
    chunks: u64,
    samples: u64,
    chunk: &[f32],
) -> Result<(), String> {
    let peak = chunk
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    let rms = if chunk.is_empty() {
        0.0
    } else {
        let mean_square =
            chunk.iter().map(|sample| sample * sample).sum::<f32>() / chunk.len() as f32;
        mean_square.sqrt()
    };

    app.emit(
        "audio-capture-stats",
        AudioCaptureStatsEvent {
            source: "process".to_string(),
            chunks,
            samples,
            rms,
            peak,
            silent: peak < 0.0005,
        },
    )
    .map_err(|e| e.to_string())
}

#[cfg(windows)]
mod windows_process_loopback {
    use super::{emit_audio_capture_stats, emit_audio_chunk, process_chunk};
    use std::mem::{size_of, ManuallyDrop};
    use std::ptr::null_mut;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use tauri::AppHandle;
    use windows::Win32::Media::Audio::IActivateAudioInterfaceCompletionHandler_Impl;
    use windows::{
        core::{implement, Interface, GUID},
        Win32::{
            Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0},
            Media::{
                Audio::{
                    ActivateAudioInterfaceAsync, IActivateAudioInterfaceAsyncOperation,
                    IActivateAudioInterfaceCompletionHandler, IAudioCaptureClient, IAudioClient,
                    AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY, AUDCLNT_BUFFERFLAGS_SILENT,
                    AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
                    AUDCLNT_STREAMFLAGS_EVENTCALLBACK, AUDCLNT_STREAMFLAGS_LOOPBACK,
                    AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, AUDIOCLIENT_ACTIVATION_PARAMS,
                    AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
                    AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS,
                    PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
                    VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, WAVEFORMATEX, WAVEFORMATEXTENSIBLE,
                    WAVE_FORMAT_PCM,
                },
                Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
            },
            System::{
                Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED},
                Threading::{CreateEventW, SetEvent, WaitForSingleObject},
            },
        },
    };
    use windows_core::imp;
    use windows_core::PROPVARIANT;

    const REFTIMES_PER_SEC: i64 = 10_000_000;
    const CAPTURE_BUFFER_MS: i64 = 100;
    const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
    const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
    const KSDATAFORMAT_SUBTYPE_PCM: GUID = GUID::from_u128(0x00000001_0000_0010_8000_00aa00389b71);

    struct OwnedEvent(HANDLE);

    impl OwnedEvent {
        fn create() -> Result<Self, String> {
            unsafe { CreateEventW(None, false, false, None) }
                .map(Self)
                .map_err(|e| e.to_string())
        }

        fn raw(&self) -> HANDLE {
            self.0
        }
    }

    impl Drop for OwnedEvent {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }

    pub fn run(app: AppHandle, pid: u32, stop: Arc<AtomicBool>) -> Result<(), String> {
        capture_process_audio_chunks(pid, stop, |chunk| process_chunk(&app, chunk))
    }

    pub fn run_raw(
        app: AppHandle,
        pid: u32,
        stop: Arc<AtomicBool>,
        persist_audio: bool,
    ) -> Result<(), String> {
        let mut chunks = 0_u64;
        let mut samples = 0_u64;

        capture_process_audio_chunks(pid, stop, |chunk| {
            chunks += 1;
            samples += chunk.len() as u64;

            emit_audio_chunk(&app, chunk, persist_audio)?;
            emit_audio_capture_stats(&app, chunks, samples, chunk)
        })
    }

    fn capture_process_audio_chunks<F>(
        pid: u32,
        stop: Arc<AtomicBool>,
        mut on_chunk: F,
    ) -> Result<(), String>
    where
        F: FnMut(&[f32]) -> Result<(), String>,
    {
        unsafe {
            let hr = CoInitializeEx(None, COINIT_MULTITHREADED);
            if hr.is_err() {
                return Err(windows::core::Error::from_hresult(hr).to_string());
            }
        }

        let result = unsafe { run_inner(pid, stop, &mut on_chunk) };

        unsafe {
            CoUninitialize();
        }

        result
    }

    unsafe fn run_inner<F>(pid: u32, stop: Arc<AtomicBool>, on_chunk: &mut F) -> Result<(), String>
    where
        F: FnMut(&[f32]) -> Result<(), String>,
    {
        let audio_client = activate_process_loopback_client(pid)?;

        // Process-loopback virtual devices do not reliably implement GetMixFormat.
        // Request PCM and let the shared audio engine convert the render stream.
        let capture_format = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_PCM as u16,
            nChannels: 2,
            nSamplesPerSec: 48_000,
            nAvgBytesPerSec: 48_000 * 2 * 2,
            nBlockAlign: 2 * 2,
            wBitsPerSample: 16,
            cbSize: 0,
        };
        let sample_rate = capture_format.nSamplesPerSec;
        let channels = capture_format.nChannels as usize;
        let stream_flags = AUDCLNT_STREAMFLAGS_LOOPBACK
            | AUDCLNT_STREAMFLAGS_EVENTCALLBACK
            | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
            | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;

        audio_client
            .Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                stream_flags,
                CAPTURE_BUFFER_MS * REFTIMES_PER_SEC / 1000,
                0,
                &capture_format,
                None,
            )
            .map_err(|e| e.to_string())?;

        let event = OwnedEvent::create()?;
        audio_client
            .SetEventHandle(event.raw())
            .map_err(|e| e.to_string())?;

        let capture_client: IAudioCaptureClient =
            audio_client.GetService().map_err(|e| e.to_string())?;
        audio_client.Start().map_err(|e| e.to_string())?;

        let mut pending_16k = Vec::<f32>::new();
        let mut resampler = LinearResampler::new(sample_rate, 16_000);

        while !stop.load(Ordering::SeqCst) {
            let wait = WaitForSingleObject(event.raw(), 100);
            if wait != WAIT_OBJECT_0 {
                continue;
            }

            loop {
                let packet_frames = capture_client
                    .GetNextPacketSize()
                    .map_err(|e| e.to_string())?;

                if packet_frames == 0 {
                    break;
                }

                let mut data = null_mut();
                let mut frames = 0;
                let mut flags = 0;
                capture_client
                    .GetBuffer(&mut data, &mut frames, &mut flags, None, None)
                    .map_err(|e| e.to_string())?;

                let mono = if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
                    vec![0.0; frames as usize]
                } else {
                    read_mono_samples(data, frames as usize, channels, &capture_format)?
                };

                capture_client
                    .ReleaseBuffer(frames)
                    .map_err(|e| e.to_string())?;

                if flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32 != 0 {
                    pending_16k.clear();
                    resampler.reset();
                }

                pending_16k.extend(resampler.process(&mono));
                while pending_16k.len() >= 1024 {
                    let chunk: Vec<f32> = pending_16k.drain(..1024).collect();
                    on_chunk(&chunk)?;
                }
            }
        }

        audio_client.Stop().map_err(|e| e.to_string())?;
        Ok(())
    }

    unsafe fn activate_process_loopback_client(pid: u32) -> Result<IAudioClient, String> {
        let complete_event = OwnedEvent::create()?;
        let completion_state = Arc::new(CompletionState::default());
        let handler: IActivateAudioInterfaceCompletionHandler =
            CompletionHandler::new(complete_event.raw(), completion_state.clone()).into();

        let mut activation_params = AUDIOCLIENT_ACTIVATION_PARAMS {
            ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
            Anonymous: Default::default(),
        };

        activation_params.Anonymous.ProcessLoopbackParams = AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
            TargetProcessId: pid,
            ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
        };

        let propvariant = ManuallyDrop::new(PROPVARIANT::from_raw(imp::PROPVARIANT {
            Anonymous: imp::PROPVARIANT_0 {
                Anonymous: imp::PROPVARIANT_0_0 {
                    vt: windows::Win32::System::Variant::VT_BLOB.0,
                    wReserved1: 0,
                    wReserved2: 0,
                    wReserved3: 0,
                    Anonymous: imp::PROPVARIANT_0_0_0 {
                        blob: imp::BLOB {
                            cbSize: size_of::<AUDIOCLIENT_ACTIVATION_PARAMS>() as u32,
                            pBlobData: &mut activation_params as *mut _ as *mut u8,
                        },
                    },
                },
            },
        }));

        let _operation = ActivateAudioInterfaceAsync(
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            &IAudioClient::IID,
            Some(propvariant.as_raw() as *const _ as *const _),
            &handler,
        )
        .map_err(|e| e.to_string())?;

        WaitForSingleObject(complete_event.raw(), u32::MAX);

        completion_state.audio_client()
    }

    #[derive(Default)]
    struct CompletionState {
        audio_client: std::sync::Mutex<Option<IAudioClient>>,
        error: std::sync::Mutex<Option<String>>,
    }

    impl CompletionState {
        fn audio_client(&self) -> Result<IAudioClient, String> {
            if let Some(error) = self.error.lock().unwrap().take() {
                return Err(error);
            }
            self.audio_client
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| "Audio client activation completed without a client.".to_string())
        }
    }

    #[implement(IActivateAudioInterfaceCompletionHandler)]
    struct CompletionHandler {
        complete_event: HANDLE,
        state: Arc<CompletionState>,
    }

    impl CompletionHandler {
        fn new(complete_event: HANDLE, state: Arc<CompletionState>) -> Self {
            Self {
                complete_event,
                state,
            }
        }
    }

    #[allow(non_snake_case)]
    impl IActivateAudioInterfaceCompletionHandler_Impl for CompletionHandler_Impl {
        fn ActivateCompleted(
            &self,
            operation: Option<&IActivateAudioInterfaceAsyncOperation>,
        ) -> windows::core::Result<()> {
            unsafe {
                if let Some(operation) = operation {
                    let mut activate_result = windows::core::HRESULT(0);
                    let mut activated_interface = None;
                    operation.GetActivateResult(&mut activate_result, &mut activated_interface)?;

                    if activate_result.is_ok() {
                        match activated_interface
                            .ok_or_else(windows::core::Error::from_win32)?
                            .cast::<IAudioClient>()
                        {
                            Ok(client) => {
                                *self.state.audio_client.lock().unwrap() = Some(client);
                            }
                            Err(error) => {
                                *self.state.error.lock().unwrap() = Some(error.to_string());
                            }
                        }
                    } else {
                        *self.state.error.lock().unwrap() =
                            Some(windows::core::Error::from(activate_result).to_string());
                    }
                } else {
                    *self.state.error.lock().unwrap() =
                        Some("ActivateCompleted was called without an operation.".to_string());
                }

                let _ = SetEvent(self.complete_event);
            }

            Ok(())
        }
    }

    unsafe fn read_mono_samples(
        data: *mut u8,
        frames: usize,
        channels: usize,
        format: &WAVEFORMATEX,
    ) -> Result<Vec<f32>, String> {
        let format_tag = std::ptr::addr_of!(format.wFormatTag).read_unaligned();
        let bits_per_sample = std::ptr::addr_of!(format.wBitsPerSample).read_unaligned();

        match format_tag {
            WAVE_FORMAT_IEEE_FLOAT => {
                let samples = std::slice::from_raw_parts(data as *const f32, frames * channels);
                Ok(to_mono(samples, frames, channels))
            }
            tag if tag == WAVE_FORMAT_PCM as u16 => match bits_per_sample {
                16 => {
                    let samples = std::slice::from_raw_parts(data as *const i16, frames * channels);
                    Ok(to_mono_i16(samples, frames, channels))
                }
                32 => {
                    let samples = std::slice::from_raw_parts(data as *const i32, frames * channels);
                    Ok(to_mono_i32(samples, frames, channels))
                }
                bits => Err(format!("Unsupported PCM bit depth: {}", bits)),
            },
            WAVE_FORMAT_EXTENSIBLE => read_extensible_mono_samples(data, frames, channels, format),
            other => Err(format!("Unsupported WASAPI format tag: {}", other)),
        }
    }

    unsafe fn read_extensible_mono_samples(
        data: *mut u8,
        frames: usize,
        channels: usize,
        format: &WAVEFORMATEX,
    ) -> Result<Vec<f32>, String> {
        let extensible = format as *const _ as *const WAVEFORMATEXTENSIBLE;
        let sub_format = std::ptr::addr_of!((*extensible).SubFormat).read_unaligned();
        let bits_per_sample = std::ptr::addr_of!(format.wBitsPerSample).read_unaligned();

        if sub_format == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT {
            let samples = std::slice::from_raw_parts(data as *const f32, frames * channels);
            Ok(to_mono(samples, frames, channels))
        } else if sub_format == KSDATAFORMAT_SUBTYPE_PCM {
            match bits_per_sample {
                16 => {
                    let samples = std::slice::from_raw_parts(data as *const i16, frames * channels);
                    Ok(to_mono_i16(samples, frames, channels))
                }
                32 => {
                    let samples = std::slice::from_raw_parts(data as *const i32, frames * channels);
                    Ok(to_mono_i32(samples, frames, channels))
                }
                bits => Err(format!("Unsupported extensible PCM bit depth: {}", bits)),
            }
        } else {
            Err("Unsupported WAVE_FORMAT_EXTENSIBLE subtype".to_string())
        }
    }

    fn to_mono(samples: &[f32], frames: usize, channels: usize) -> Vec<f32> {
        samples
            .chunks_exact(channels)
            .take(frames)
            .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
            .collect()
    }

    fn to_mono_i16(samples: &[i16], frames: usize, channels: usize) -> Vec<f32> {
        samples
            .chunks_exact(channels)
            .take(frames)
            .map(|frame| {
                frame
                    .iter()
                    .map(|sample| *sample as f32 / i16::MAX as f32)
                    .sum::<f32>()
                    / channels as f32
            })
            .collect()
    }

    fn to_mono_i32(samples: &[i32], frames: usize, channels: usize) -> Vec<f32> {
        samples
            .chunks_exact(channels)
            .take(frames)
            .map(|frame| {
                frame
                    .iter()
                    .map(|sample| *sample as f32 / i32::MAX as f32)
                    .sum::<f32>()
                    / channels as f32
            })
            .collect()
    }

    struct LinearResampler {
        from: u32,
        to: u32,
        position: f64,
        previous: f32,
        has_previous: bool,
    }

    impl LinearResampler {
        fn new(from: u32, to: u32) -> Self {
            Self {
                from,
                to,
                position: 0.0,
                previous: 0.0,
                has_previous: false,
            }
        }

        fn reset(&mut self) {
            self.position = 0.0;
            self.previous = 0.0;
            self.has_previous = false;
        }

        fn process(&mut self, input: &[f32]) -> Vec<f32> {
            if input.is_empty() {
                return Vec::new();
            }
            if self.from == self.to {
                self.previous = *input.last().unwrap_or(&self.previous);
                self.has_previous = true;
                return input.to_vec();
            }

            let step = self.from as f64 / self.to as f64;
            let mut extended = Vec::with_capacity(input.len() + 1);
            extended.push(if self.has_previous {
                self.previous
            } else {
                input[0]
            });
            extended.extend_from_slice(input);

            let mut output = Vec::with_capacity((input.len() as f64 / step).ceil() as usize);
            while self.position + 1.0 < extended.len() as f64 {
                let idx = self.position.floor() as usize;
                let frac = (self.position - idx as f64) as f32;
                let a = extended[idx];
                let b = extended[idx + 1];
                output.push(a + (b - a) * frac);
                self.position += step;
            }

            self.position -= input.len() as f64;
            self.previous = *input.last().unwrap_or(&self.previous);
            self.has_previous = true;
            output
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::fs;
        use std::process::{Command, Stdio};
        use std::sync::Mutex;
        use std::thread;
        use std::time::{Duration, Instant};

        #[test]
        fn mixes_interleaved_float_frames_to_mono() {
            let stereo = [1.0_f32, -1.0, 0.5, 0.25, -0.5, -0.25];
            let mono = to_mono(&stereo, 3, 2);

            assert_eq!(mono, vec![0.0, 0.375, -0.375]);
        }

        #[test]
        fn converts_i16_frames_to_normalized_mono() {
            let stereo = [i16::MAX, i16::MAX, i16::MIN, i16::MIN];
            let mono = to_mono_i16(&stereo, 2, 2);

            assert!((mono[0] - 1.0).abs() < 0.0001);
            assert!(mono[1] < -1.0);
        }

        #[test]
        fn resampler_downsamples_48khz_to_16khz() {
            let input = vec![0.5_f32; 48_000];
            let mut resampler = LinearResampler::new(48_000, 16_000);
            let output = resampler.process(&input);

            assert!((output.len() as i32 - 16_000).abs() <= 1);
            assert!(output.iter().all(|sample| (*sample - 0.5).abs() < 0.0001));
        }

        #[test]
        #[ignore = "Requires Windows process loopback support and an active audio output device."]
        fn captures_only_the_target_process_audio() {
            let target_wav = std::env::temp_dir().join("tauri_llm_target_440hz.wav");
            let distractor_wav = std::env::temp_dir().join("tauri_llm_distractor_1000hz.wav");
            write_sine_wav(&target_wav, 440.0, 5.0, 48_000);
            write_sine_wav(&distractor_wav, 1_000.0, 5.0, 48_000);

            let mut target = Command::new("powershell")
                .args(["-NoProfile", "-Command", &sound_player_script(&target_wav)])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("failed to start target audio process");
            let mut distractor = Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    &sound_player_script(&distractor_wav),
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("failed to start distractor audio process");

            let target_pid = target.id();
            let stop = Arc::new(AtomicBool::new(false));
            let samples = Arc::new(Mutex::new(Vec::<f32>::new()));
            let samples_for_worker = samples.clone();
            let stop_for_worker = stop.clone();

            let worker = thread::spawn(move || {
                capture_process_audio_chunks(target_pid, stop_for_worker, |chunk| {
                    samples_for_worker.lock().unwrap().extend_from_slice(chunk);
                    Ok(())
                })
            });

            let deadline = Instant::now() + Duration::from_secs(7);
            while Instant::now() < deadline {
                thread::sleep(Duration::from_millis(100));
                let captured = samples.lock().unwrap();
                if captured.len() > 32_000 && rms(&captured) > 0.005 {
                    break;
                }
            }

            stop.store(true, Ordering::SeqCst);
            let _ = target.kill();
            let _ = target.wait();
            let _ = distractor.kill();
            let _ = distractor.wait();
            let result = worker.join().expect("capture worker panicked");
            let _ = fs::remove_file(&target_wav);
            let _ = fs::remove_file(&distractor_wav);

            if let Err(error) = result {
                if error.contains("0x80004001") {
                    eprintln!(
                        "Skipping process loopback capture assertion: Application Loopback is not implemented in this Windows audio environment ({error})."
                    );
                    return;
                }

                panic!("process loopback capture failed: {error}");
            }

            let captured = samples.lock().unwrap();
            assert!(
                captured.len() > 32_000,
                "captured too few samples: {}",
                captured.len()
            );
            assert!(rms(&captured) > 0.005, "captured signal was silent");

            let target_amplitude = tone_amplitude(&captured, 440.0, 16_000.0);
            let distractor_amplitude = tone_amplitude(&captured, 1_000.0, 16_000.0);
            eprintln!(
                "captured {} samples, rms {:.6}, target 440 Hz {:.6}, distractor 1000 Hz {:.6}, isolation ratio {:.1}x",
                captured.len(),
                rms(&captured),
                target_amplitude,
                distractor_amplitude,
                target_amplitude / distractor_amplitude.max(f64::EPSILON)
            );
            assert!(
                target_amplitude > 0.01,
                "target tone was too weak: {target_amplitude:.6}"
            );
            assert!(
                target_amplitude > distractor_amplitude * 8.0,
                "process isolation failed: target amplitude {target_amplitude:.6}, distractor amplitude {distractor_amplitude:.6}"
            );
        }

        fn sound_player_script(path: &std::path::Path) -> String {
            format!(
                "$player = New-Object System.Media.SoundPlayer '{}'; Start-Sleep -Milliseconds 600; $player.PlaySync()",
                path.display().to_string().replace('\'', "''")
            )
        }

        fn tone_amplitude(samples: &[f32], frequency: f64, sample_rate: f64) -> f64 {
            if samples.len() < 2 {
                return 0.0;
            }

            let denominator = (samples.len() - 1) as f64;
            let mut sin_sum = 0.0;
            let mut cos_sum = 0.0;
            let mut window_sum = 0.0;

            for (index, sample) in samples.iter().enumerate() {
                let phase = std::f64::consts::TAU * frequency * index as f64 / sample_rate;
                let window = 0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / denominator).cos();
                let value = *sample as f64 * window;
                sin_sum += value * phase.sin();
                cos_sum += value * phase.cos();
                window_sum += window;
            }

            2.0 * sin_sum.hypot(cos_sum) / window_sum
        }

        fn rms(samples: &[f32]) -> f32 {
            if samples.is_empty() {
                return 0.0;
            }

            let mean_square =
                samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32;
            mean_square.sqrt()
        }

        fn write_sine_wav(path: &std::path::Path, frequency: f32, seconds: f32, sample_rate: u32) {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(path, spec).unwrap();
            let sample_count = (seconds * sample_rate as f32) as usize;

            for index in 0..sample_count {
                let t = index as f32 / sample_rate as f32;
                let value = (t * frequency * std::f32::consts::TAU).sin() * 0.4;
                writer
                    .write_sample((value * i16::MAX as f32) as i16)
                    .unwrap();
            }

            writer.finalize().unwrap();
        }
    }
}
