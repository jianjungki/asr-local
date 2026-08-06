// Debug logging macro controlled by TAURI_LLM_DEBUG environment variable
macro_rules! debug_log {
    ($($arg:tt)*) => {
        {
            let env_val = std::env::var("TAURI_LLM_DEBUG").unwrap_or_else(|_| String::from("not_set"));
            let is_enabled = env_val == "1" || env_val.to_lowercase() == "true";

            // Always log to stderr for debugging (can be redirected)
            if is_enabled {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                eprintln!("[DEBUG][{}] {}", timestamp, format!($($arg)*));
            }
        }
    };
}

use serde::{Deserialize, Serialize};
mod assistant_orchestrator;
mod audio_capture;
mod inference;
mod model_runtime;
mod recording_store;
mod speaker_diarization;
mod vision;
use assistant_orchestrator::{AssistantModelSet, AssistantRequest, AssistantState};
use audio_capture::AudioCaptureState;
use inference::AppState;
use model_runtime::{ConversationMessage, GenAiModelRuntime, ProviderTarget};
use recording_store::RecordingStoreState;
use speaker_diarization::SpeakerDiarizationState;
use std::collections::HashMap;
use std::process::{Command as StdCommand, Stdio};
use tauri::{command, AppHandle, Emitter, Manager, Runtime, State, Window};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};
use tauri_plugin_store::StoreExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

const SENSE_VOICE_MODEL_DIR: &str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8";
const SENSE_VOICE_MODEL_REPO: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17";
const PARAFORMER_STREAMING_MODEL_DIR: &str = "sherpa-onnx-streaming-paraformer-bilingual-zh-en";
const PARAFORMER_STREAMING_ARCHIVE_URLS: [&str; 2] = [
    "https://gh-proxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2",
];
const ZIPFORMER_ZH_MODEL_DIR: &str = "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03";
const ZIPFORMER_ZH_MODEL_REPO: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03";
const SILERO_VAD_MODEL_DIR: &str = "silero-vad";
const SILERO_VAD_MODEL_FILE: &str = "silero_vad.onnx";
const SILERO_VAD_MODEL_URLS: [&str; 2] = [
    "https://gh-proxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
];
const PUNCTUATION_MODEL_DIR: &str =
    "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8";
const PUNCTUATION_MODEL_FILE: &str = "model.int8.onnx";
const PUNCTUATION_MODEL_URLS: [&str; 2] = [
    "https://gh-proxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2",
];

struct LocalModelSpec {
    id: &'static str,
    directory: &'static str,
    repository: Option<&'static str>,
}

fn local_model_spec(model_id: &str) -> Result<LocalModelSpec, String> {
    match model_id.trim() {
        "" | inference::PARAFORMER_STREAMING_MODEL_ID => Ok(LocalModelSpec {
            id: inference::PARAFORMER_STREAMING_MODEL_ID,
            directory: PARAFORMER_STREAMING_MODEL_DIR,
            repository: None,
        }),
        inference::SENSE_VOICE_MODEL_ID => Ok(LocalModelSpec {
            id: inference::SENSE_VOICE_MODEL_ID,
            directory: SENSE_VOICE_MODEL_DIR,
            repository: Some(SENSE_VOICE_MODEL_REPO),
        }),
        inference::ZIPFORMER_ZH_MODEL_ID => Ok(LocalModelSpec {
            id: inference::ZIPFORMER_ZH_MODEL_ID,
            directory: ZIPFORMER_ZH_MODEL_DIR,
            repository: Some(ZIPFORMER_ZH_MODEL_REPO),
        }),
        other => Err(format!("不支持的本地语音模型：{other}")),
    }
}

fn local_model_path(app: &AppHandle, spec: &LocalModelSpec) -> Result<std::path::PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("models")
        .join(spec.directory))
}

fn silero_vad_model_path(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("models")
        .join(SILERO_VAD_MODEL_DIR)
        .join(SILERO_VAD_MODEL_FILE))
}

fn punctuation_model_path(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("models")
        .join(PUNCTUATION_MODEL_DIR)
        .join(PUNCTUATION_MODEL_FILE))
}

async fn ensure_silero_vad_model(
    app: &AppHandle,
    state: &AppState,
) -> Result<std::path::PathBuf, String> {
    let model_path = silero_vad_model_path(app)?;
    if inference::vad_model_file_is_ready(&model_path) {
        return Ok(model_path);
    }

    let _download_guard = state.vad_download.lock().await;
    if inference::vad_model_file_is_ready(&model_path) {
        return Ok(model_path);
    }

    let model_dir = model_path
        .parent()
        .ok_or_else(|| "Silero VAD 模型目录无效".to_string())?;
    tokio::fs::create_dir_all(model_dir)
        .await
        .map_err(|error| format!("无法创建 Silero VAD 模型目录：{error}"))?;

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|error| format!("创建 Silero VAD 下载客户端失败：{error}"))?;
    let mut model_bytes = None;
    let mut last_error = String::new();
    for url in SILERO_VAD_MODEL_URLS {
        match client.get(url).send().await {
            Ok(response) => match response.error_for_status() {
                Ok(response) => match response.bytes().await {
                    Ok(bytes) if bytes.len() > 512 * 1024 => {
                        model_bytes = Some(bytes);
                        break;
                    }
                    Ok(_) => last_error = "下载文件不完整".to_string(),
                    Err(error) => last_error = error.to_string(),
                },
                Err(error) => last_error = error.to_string(),
            },
            Err(error) => last_error = error.to_string(),
        }
    }
    let bytes = model_bytes.ok_or_else(|| format!("下载 Silero VAD 失败：{last_error}"))?;

    let partial_path = model_path.with_extension("onnx.part");
    tokio::fs::write(&partial_path, &bytes)
        .await
        .map_err(|error| format!("保存 Silero VAD 模型失败：{error}"))?;
    if model_path.exists() {
        tokio::fs::remove_file(&model_path)
            .await
            .map_err(|error| format!("替换 Silero VAD 模型失败：{error}"))?;
    }
    tokio::fs::rename(&partial_path, &model_path)
        .await
        .map_err(|error| format!("完成 Silero VAD 模型缓存失败：{error}"))?;

    Ok(model_path)
}

async fn ensure_punctuation_model(
    app: &AppHandle,
    state: &AppState,
) -> Result<std::path::PathBuf, String> {
    let model_path = punctuation_model_path(app)?;
    if inference::punctuation_model_file_is_ready(&model_path) {
        return Ok(model_path);
    }

    let _download_guard = state.punctuation_download.lock().await;
    if inference::punctuation_model_file_is_ready(&model_path) {
        return Ok(model_path);
    }

    let model_dir = model_path
        .parent()
        .ok_or_else(|| "标点模型目录无效".to_string())?;
    let models_dir = model_dir
        .parent()
        .ok_or_else(|| "模型缓存目录无效".to_string())?;
    tokio::fs::create_dir_all(models_dir)
        .await
        .map_err(|error| format!("无法创建标点模型目录：{error}"))?;

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|error| format!("创建标点模型下载客户端失败：{error}"))?;
    let mut archive_bytes = None;
    let mut last_error = String::new();
    for url in PUNCTUATION_MODEL_URLS {
        match client.get(url).send().await {
            Ok(response) => match response.error_for_status() {
                Ok(response) => match response.bytes().await {
                    Ok(bytes) if bytes.len() > 50 * 1024 * 1024 => {
                        archive_bytes = Some(bytes);
                        break;
                    }
                    Ok(_) => last_error = "下载文件不完整".to_string(),
                    Err(error) => last_error = error.to_string(),
                },
                Err(error) => last_error = error.to_string(),
            },
            Err(error) => last_error = error.to_string(),
        }
    }
    let bytes = archive_bytes.ok_or_else(|| format!("下载标点模型失败：{last_error}"))?;
    let archive_path = models_dir.join(format!("{PUNCTUATION_MODEL_DIR}.tar.bz2.part"));
    tokio::fs::write(&archive_path, &bytes)
        .await
        .map_err(|error| format!("保存标点模型压缩包失败：{error}"))?;
    drop(bytes);

    let output = Command::new("tar")
        .arg("-xjf")
        .arg(&archive_path)
        .arg("-C")
        .arg(models_dir)
        .output()
        .await
        .map_err(|error| format!("无法启动标点模型解压程序：{error}"))?;
    if !output.status.success() {
        return Err(format!(
            "解压标点模型失败：{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let _ = tokio::fs::remove_file(&archive_path).await;

    if !inference::punctuation_model_file_is_ready(&model_path) {
        return Err("解压后的标点模型文件不完整".to_string());
    }
    Ok(model_path)
}

async fn download_file_with_progress(
    client: &reqwest::Client,
    window: &Window,
    url: &str,
    destination: &std::path::Path,
) -> Result<u64, String> {
    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?;
    let total_bytes = response.content_length();
    let mut file = tokio::fs::File::create(destination)
        .await
        .map_err(|error| format!("创建模型下载文件失败：{error}"))?;
    let mut downloaded_bytes = 0_u64;
    let mut last_percent = u64::MAX;

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| format!("接收模型数据失败：{error}"))?
    {
        file.write_all(&chunk)
            .await
            .map_err(|error| format!("写入模型数据失败：{error}"))?;
        downloaded_bytes += chunk.len() as u64;

        if let Some(total_bytes) = total_bytes.filter(|total| *total > 0) {
            // Keep 100% for extraction and support-model validation completion.
            let percent = (downloaded_bytes.saturating_mul(100) / total_bytes).min(99);
            if percent != last_percent {
                last_percent = percent;
                window
                    .emit("download-progress", format!("{percent}%"))
                    .unwrap_or(());
                if percent % 5 == 0 {
                    window
                        .emit(
                            "download-log",
                            format!(
                                "Paraformer 下载 {percent}%（{:.0}/{:.0} MB）",
                                downloaded_bytes as f64 / 1_048_576.0,
                                total_bytes as f64 / 1_048_576.0
                            ),
                        )
                        .unwrap_or(());
                }
            }
        }
    }

    file.flush()
        .await
        .map_err(|error| format!("刷新模型下载文件失败：{error}"))?;
    Ok(downloaded_bytes)
}

async fn download_paraformer_streaming_model(
    window: &Window,
    model_dir: &std::path::Path,
) -> Result<(), String> {
    let models_dir = model_dir
        .parent()
        .ok_or_else(|| "模型缓存目录无效".to_string())?;
    tokio::fs::create_dir_all(models_dir)
        .await
        .map_err(|error| format!("无法创建模型缓存目录：{error}"))?;
    let archive_path = models_dir.join(format!("{PARAFORMER_STREAMING_MODEL_DIR}.tar.bz2.part"));
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(std::time::Duration::from_secs(3_600))
        .build()
        .map_err(|error| format!("创建 Paraformer 下载客户端失败：{error}"))?;

    let mut last_error = String::new();
    let mut downloaded = false;
    for url in PARAFORMER_STREAMING_ARCHIVE_URLS {
        let _ = tokio::fs::remove_file(&archive_path).await;
        window
            .emit("download-log", format!("正在下载 Paraformer：{url}"))
            .unwrap_or(());
        match download_file_with_progress(&client, window, url, &archive_path).await {
            Ok(bytes) if bytes > 500 * 1024 * 1024 => {
                downloaded = true;
                break;
            }
            Ok(_) => last_error = "下载文件不完整".to_string(),
            Err(error) => last_error = error,
        }
    }
    if !downloaded {
        let _ = tokio::fs::remove_file(&archive_path).await;
        return Err(format!("下载 Paraformer 模型失败：{last_error}"));
    }

    window
        .emit("download-log", "正在解压 Paraformer int8 模型...")
        .unwrap_or(());
    let output = Command::new("tar")
        .arg("-xjf")
        .arg(&archive_path)
        .arg("-C")
        .arg(models_dir)
        .output()
        .await
        .map_err(|error| format!("无法启动 Paraformer 解压程序：{error}"))?;
    let _ = tokio::fs::remove_file(&archive_path).await;
    if !output.status.success() {
        return Err(format!(
            "解压 Paraformer 模型失败：{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    if !inference::model_files_exist(model_dir, inference::PARAFORMER_STREAMING_MODEL_ID) {
        return Err("解压后的 Paraformer Streaming 模型文件不完整".to_string());
    }

    // The release also contains fp32 copies; the app uses int8 only.
    for file_name in ["encoder.onnx", "decoder.onnx"] {
        let _ = tokio::fs::remove_file(model_dir.join(file_name)).await;
    }
    Ok(())
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default, rename_all = "camelCase")]
pub struct ProviderConfig {
    base_url: String,
    api_key: String,
    text_model: String,
    vision_model: String,
    available_models: Vec<String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: String::new(),
            text_model: String::new(),
            vision_model: String::new(),
            available_models: Vec::new(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    text: String,
    always_on_top: bool,
    font_size: u32,
    font_color: String,
    background_color: String,
    opacity: f64,
    caption_position: String,
    caption_background_style: String,
    translation_enabled: bool,
    translation_source_language: String,
    translation_target_language: String,
    model_type: String,
    audio_source: String,
    #[serde(default)]
    audio_input_device_id: String,
    process_audio_pid: u32,
    local_model_path: String,
    is_model_downloaded: bool,
    local_speech_model: String,
    sense_voice_language: String,
    assistant_provider: String,
    assistant_text_model: String,
    assistant_vision_model: String,
    ollama_base_url: String,
    speech_provider: String,
    assistant_text_provider: String,
    assistant_vision_provider: String,
    provider_configs: HashMap<String, ProviderConfig>,
    auto_format_transcripts: bool,
    speaker_diarization_enabled: bool,
    toggle_recording_hotkey: String,
    capture_visual_hotkey: String,
    #[serde(default)]
    start_recording_hotkey: String,
    #[serde(default)]
    stop_recording_hotkey: String,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AudioProcessTarget {
    pid: u32,
    name: String,
    window_title: Option<String>,
    executable_path: Option<String>,
    command_line: Option<String>,
    session_id: Option<u32>,
    #[serde(default)]
    has_audio_session: bool,
    #[serde(default)]
    audio_session_count: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            text: "默认字幕".to_string(),
            always_on_top: true,
            font_size: 24,
            font_color: "#FFFFFF".to_string(),
            background_color: "#000000".to_string(),
            opacity: 0.5,
            caption_position: "bottom".to_string(),
            caption_background_style: "glass".to_string(),
            translation_enabled: false,
            translation_source_language: "auto".to_string(),
            translation_target_language: "zh-CN".to_string(),
            model_type: "online".to_string(),
            audio_source: "microphone".to_string(),
            audio_input_device_id: "".to_string(),
            process_audio_pid: 0,
            local_model_path: "".to_string(),
            is_model_downloaded: false,
            local_speech_model: inference::DEFAULT_LOCAL_SPEECH_MODEL_ID.to_string(),
            sense_voice_language: inference::DEFAULT_SENSE_VOICE_LANGUAGE.to_string(),
            assistant_provider: "dashscope".to_string(),
            assistant_text_model: "qwen-plus".to_string(),
            assistant_vision_model: "qwen-vl-plus".to_string(),
            ollama_base_url: "http://127.0.0.1:11434".to_string(),
            // Empty values let the frontend distinguish an old settings file
            // and migrate its former shared provider fields once.
            speech_provider: String::new(),
            assistant_text_provider: String::new(),
            assistant_vision_provider: String::new(),
            provider_configs: HashMap::new(),
            auto_format_transcripts: false,
            speaker_diarization_enabled: true,
            toggle_recording_hotkey: "Alt+F1".to_string(),
            capture_visual_hotkey: "Alt+F2".to_string(),
            start_recording_hotkey: String::new(),
            stop_recording_hotkey: String::new(),
        }
    }
}

#[command]
fn list_visual_windows() -> Result<Vec<vision::VisualWindowTarget>, String> {
    vision::list_windows()
}

#[command]
async fn capture_visual_window(
    window_id: Option<String>,
) -> Result<vision::CapturedWindow, String> {
    tauri::async_runtime::spawn_blocking(move || vision::capture_window(window_id.as_deref()))
        .await
        .map_err(|error| format!("窗口截图任务失败：{error}"))?
}

#[command]
async fn chat_with_assistant(
    state: State<'_, AssistantState>,
    models: AssistantModelSet,
    system_prompt: Option<String>,
    messages: Vec<ConversationMessage>,
    image_data_url: Option<String>,
) -> Result<String, String> {
    let image_data_url = image_data_url
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    state
        .respond(AssistantRequest {
            models,
            system_prompt,
            messages,
            image_data_url,
            allow_desktop_actions: false,
        })
        .await
}

#[command]
async fn list_provider_models(provider: ProviderTarget) -> Result<Vec<String>, String> {
    GenAiModelRuntime::default().list_models(provider).await
}

fn translation_language_name(code: &str, allow_auto: bool) -> Result<&'static str, String> {
    match code {
        "auto" if allow_auto => Ok("automatically detected language"),
        "zh-CN" => Ok("Simplified Chinese"),
        "en" => Ok("English"),
        "ja" => Ok("Japanese"),
        "ko" => Ok("Korean"),
        "es" => Ok("Spanish"),
        "fr" => Ok("French"),
        "de" => Ok("German"),
        _ => Err(format!("Unsupported translation language: {code}")),
    }
}

#[command]
async fn translate_caption(
    api_key: String,
    text: String,
    source_language: String,
    target_language: String,
) -> Result<String, String> {
    let text = text.trim();
    if api_key.trim().is_empty() {
        return Err("DashScope API Key 未配置".to_string());
    }
    if text.is_empty() {
        return Ok(String::new());
    }

    let source = translation_language_name(&source_language, true)?;
    let target = translation_language_name(&target_language, false)?;
    let prompt = format!(
        "Translate the following real-time subtitle from {source} to {target}. Return only the translation. Preserve names, numbers, tone, and punctuation. Do not explain or add quotation marks.\n\n{text}"
    );

    let response = reqwest::Client::new()
        .post("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        .bearer_auth(api_key.trim())
        .json(&serde_json::json!({
            "model": "qwen-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise, concise real-time subtitle translator."
                },
                { "role": "user", "content": prompt }
            ],
            "temperature": 0.1
        }))
        .send()
        .await
        .map_err(|error| format!("翻译服务连接失败：{error}"))?;

    let status = response.status();
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("翻译服务响应无效：{error}"))?;

    if !status.is_success() {
        let message = payload
            .pointer("/error/message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("未知错误");
        return Err(format!("翻译失败（{}）：{}", status.as_u16(), message));
    }

    payload
        .pointer("/choices/0/message/content")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| "翻译服务未返回文本".to_string())
}

#[command]
fn list_audio_processes() -> Result<Vec<AudioProcessTarget>, String> {
    #[cfg(windows)]
    {
        let audio_session_counts = windows_audio_session_counts().unwrap_or_default();
        let command = r#"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$processesById = @{}
Get-Process | ForEach-Object {
  $processesById[$_.Id] = $_
}
Get-CimInstance Win32_Process | ForEach-Object {
  $process = $processesById[[int]$_.ProcessId]
  [PSCustomObject]@{
    pid = [int]$_.ProcessId
    name = if ($process) { $process.ProcessName } else { $_.Name }
    windowTitle = if ($process) { $process.MainWindowTitle } else { "" }
    executablePath = $_.ExecutablePath
    commandLine = $_.CommandLine
    sessionId = if ($_.SessionId -ne $null) { [int]$_.SessionId } else { $null }
  }
} |
  Sort-Object name, pid |
  ConvertTo-Json -Compress
"#;

        let output = StdCommand::new("powershell")
            .args(["-NoProfile", "-Command", command])
            .output()
            .map_err(|e| format!("Failed to run PowerShell process listing: {e}"))?;

        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let trimmed = stdout.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }

        let value: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("Failed to parse process list: {e}"))?;

        let mut processes = match value {
            serde_json::Value::Array(items) => items
                .into_iter()
                .filter_map(
                    |item| match serde_json::from_value::<AudioProcessTarget>(item) {
                        Ok(process) => Some(process),
                        Err(error) => {
                            eprintln!(
                                "Skipping process list item that could not be parsed: {error}"
                            );
                            None
                        }
                    },
                )
                .collect::<Vec<_>>(),
            serde_json::Value::Object(_) => {
                vec![serde_json::from_value::<AudioProcessTarget>(value)
                    .map_err(|e| format!("Failed to parse process list item: {e}"))?]
            }
            _ => Vec::new(),
        };

        for process in &mut processes {
            let count = audio_session_counts.get(&process.pid).copied().unwrap_or(0);
            process.audio_session_count = count;
            process.has_audio_session = count > 0;
        }

        processes.retain(|process| process.pid > 0 && !process.name.trim().is_empty());
        processes.sort_by(|a, b| {
            b.has_audio_session
                .cmp(&a.has_audio_session)
                .then_with(|| b.audio_session_count.cmp(&a.audio_session_count))
                .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
                .then(a.pid.cmp(&b.pid))
        });
        Ok(processes)
    }

    #[cfg(not(windows))]
    {
        Err("Process audio capture is currently implemented only on Windows.".to_string())
    }
}

#[cfg(windows)]
fn windows_audio_session_counts() -> Result<HashMap<u32, u32>, String> {
    use windows::core::Interface;
    use windows::Win32::Media::Audio::{
        eConsole, eRender, IAudioSessionControl2, IAudioSessionManager2, IMMDeviceEnumerator,
        MMDeviceEnumerator,
    };
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED,
    };

    unsafe {
        let hr = CoInitializeEx(None, COINIT_MULTITHREADED);
        let initialized = if hr.is_ok() {
            true
        } else if hr == windows::Win32::Foundation::RPC_E_CHANGED_MODE {
            false
        } else {
            return Err(windows::core::Error::from_hresult(hr).to_string());
        };

        let result = (|| -> Result<HashMap<u32, u32>, String> {
            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                    .map_err(|e| e.to_string())?;
            let device = enumerator
                .GetDefaultAudioEndpoint(eRender, eConsole)
                .map_err(|e| e.to_string())?;
            let manager: IAudioSessionManager2 = device
                .Activate(CLSCTX_ALL, None)
                .map_err(|e| e.to_string())?;
            let session_enumerator = manager.GetSessionEnumerator().map_err(|e| e.to_string())?;
            let count = session_enumerator.GetCount().map_err(|e| e.to_string())?;
            let mut sessions = HashMap::new();

            for index in 0..count {
                let session = session_enumerator
                    .GetSession(index)
                    .map_err(|e| e.to_string())?;
                let session2: IAudioSessionControl2 = session.cast().map_err(|e| e.to_string())?;
                let pid = session2.GetProcessId().map_err(|e| e.to_string())?;
                if pid > 0 {
                    *sessions.entry(pid).or_insert(0) += 1;
                }
            }

            Ok(sessions)
        })();

        if initialized {
            CoUninitialize();
        }

        result
    }
}

fn register_shortcuts<R: Runtime>(app: &AppHandle<R>, settings: &Settings) -> Result<(), String> {
    let shortcut_manager = app.global_shortcut();
    shortcut_manager
        .unregister_all()
        .map_err(|error| error.to_string())?;

    for (hotkey, event_name) in [
        (
            settings.toggle_recording_hotkey.trim(),
            "toggle_recording_shortcut",
        ),
        (
            settings.capture_visual_hotkey.trim(),
            "capture_visual_shortcut",
        ),
    ] {
        if hotkey.is_empty() {
            continue;
        }
        let Ok(shortcut) = hotkey.parse::<Shortcut>() else {
            eprintln!("Ignoring invalid global shortcut: {hotkey}");
            continue;
        };
        shortcut_manager
            .on_shortcut(shortcut, move |app, _shortcut, event| {
                if event.state == ShortcutState::Pressed {
                    let _ = app.emit(event_name, ());
                }
            })
            .map_err(|error| error.to_string())?;
    }

    Ok(())
}

#[command]
async fn save_settings<R: Runtime>(app: AppHandle<R>, settings: Settings) -> Result<(), String> {
    let store = app.store("settings.json").map_err(|e| e.to_string())?;
    store.set(
        "settings".to_string(),
        serde_json::to_value(&settings).unwrap(),
    );
    store.save().map_err(|e| e.to_string())?;
    app.emit("update-settings", &settings)
        .map_err(|e| e.to_string())?;

    register_shortcuts(&app, &settings)
}

async fn load_local_speech_model(
    app: &AppHandle,
    state: &AppState,
    model_id: &str,
    sense_voice_language: Option<&str>,
) -> Result<(), String> {
    let spec = local_model_spec(model_id)?;
    let model_dir = local_model_path(app, &spec)?;
    if !inference::model_files_exist(&model_dir, spec.id) {
        return Err(format!(
            "本地模型尚未完整下载：{}。请先在 AI 处理页面下载；若已执行下载，请确认 Git LFS 可用。",
            spec.id
        ));
    }

    let recognition_language = if spec.id == inference::SENSE_VOICE_MODEL_ID {
        inference::normalize_sense_voice_language(sense_voice_language)
            .map_err(|error| error.to_string())?
    } else {
        "zh".to_string()
    };
    let vad_model_path = Some(ensure_silero_vad_model(app, state).await?);
    let punctuation_model_path = if matches!(
        spec.id,
        inference::PARAFORMER_STREAMING_MODEL_ID | inference::ZIPFORMER_ZH_MODEL_ID
    ) {
        Some(ensure_punctuation_model(app, state).await?)
    } else {
        None
    };

    let mut model_guard = state.model.lock().await;
    if let Some(model) = model_guard.as_mut() {
        if model.model_id() == spec.id && model.recognition_language() == recognition_language {
            model.reset_session();
            return Ok(());
        }
    }

    let model_id = spec.id;
    let language = recognition_language;
    // ONNX session creation performs heavy disk and CPU work; keep it off async runtime threads.
    let model = tauri::async_runtime::spawn_blocking(move || {
        inference::SherpaOfflineModel::new(
            &model_dir,
            model_id,
            Some(&language),
            vad_model_path.as_deref(),
            punctuation_model_path.as_deref(),
            16_000,
        )
    })
    .await
    .map_err(|error| format!("Local model loading task failed: {error}"))?
    .map_err(|error| error.to_string())?;

    *model_guard = Some(model);
    Ok(())
}

#[command]
fn check_model_downloaded(app: AppHandle, model_id: String) -> Result<bool, String> {
    let spec = local_model_spec(&model_id)?;
    let model_dir = local_model_path(&app, &spec)?;
    Ok(inference::model_files_exist(&model_dir, spec.id))
}

#[command]
async fn prepare_local_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
    sense_voice_language: Option<String>,
) -> Result<(), String> {
    load_local_speech_model(
        &app,
        state.inner(),
        &model_id,
        sense_voice_language.as_deref(),
    )
    .await
}

#[command]
fn load_settings<R: Runtime>(app: AppHandle<R>) -> Result<Settings, String> {
    let store = app.store("settings.json").map_err(|e| e.to_string())?;
    match store.get("settings".to_string()) {
        Some(value) => {
            let mut settings: Settings = serde_json::from_value(value.clone()).unwrap_or_default();
            let has_toggle_hotkey = value
                .get("toggleRecordingHotkey")
                .and_then(|value| value.as_str())
                .is_some_and(|value| !value.trim().is_empty());
            if !has_toggle_hotkey && !settings.start_recording_hotkey.trim().is_empty() {
                settings.toggle_recording_hotkey = settings.start_recording_hotkey.clone();
            }
            Ok(settings)
        }
        None => {
            let default_settings = Settings::default();
            store.set(
                "settings".to_string(),
                serde_json::to_value(&default_settings).unwrap(),
            );
            store.save().map_err(|e| e.to_string())?;
            Ok(default_settings)
        }
    }
}

#[command]
async fn download_model(
    app: AppHandle,
    window: Window,
    state: State<'_, AppState>,
    model_id: String,
) -> Result<(), String> {
    debug_log!("Starting model download process");

    let spec = local_model_spec(&model_id)?;

    {
        let mut model_guard = state.model.lock().await;
        if model_guard
            .as_ref()
            .is_some_and(|model| model.model_id() == spec.id)
        {
            *model_guard = None;
        }
    }

    let model_dir = local_model_path(&app, &spec).map_err(|error| {
        debug_log!("Failed to get app data directory: {}", error);
        error
    })?;

    // Output the model directory at the start
    eprintln!("========================================");
    eprintln!("Starting model download...");
    eprintln!("Model directory: {}", model_dir.display());
    eprintln!("========================================");
    debug_log!("Model directory: {}", model_dir.display());

    // Clean up existing directory
    if model_dir.exists() {
        debug_log!("Removing existing model directory");
        std::fs::remove_dir_all(&model_dir).map_err(|e| {
            debug_log!("Failed to remove existing directory: {}", e);
            e.to_string()
        })?;
    }

    if spec.id == inference::PARAFORMER_STREAMING_MODEL_ID {
        download_paraformer_streaming_model(&window, &model_dir).await?;
    } else {
        debug_log!("Creating model directory");
        std::fs::create_dir_all(&model_dir).map_err(|e| {
            debug_log!("Failed to create directory: {}", e);
            e.to_string()
        })?;

        let repo_url = spec
            .repository
            .ok_or_else(|| "本地模型缺少下载地址".to_string())?;
        debug_log!("Repository URL: {}", repo_url);
        debug_log!("Executing git clone command");

        let mut child = Command::new("git")
            .arg("clone")
            .arg("--depth")
            .arg("1")
            .arg("--progress")
            .arg(repo_url)
            .arg(&model_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                let err_msg = format!("Failed to spawn git: {}", e);
                debug_log!("{}", err_msg);
                err_msg
            })?;

        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "Failed to capture git stderr".to_string())?;
        let mut reader = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            debug_log!("Git output: {}", line);
            if line.contains("Receiving objects:") && line.contains('%') {
                window.emit("download-progress", &line).unwrap_or(());
            }
            window.emit("download-log", &line).unwrap_or(());
        }

        let status = child.wait().await.map_err(|e| {
            debug_log!("Error waiting for git process: {}", e);
            e.to_string()
        })?;
        if !status.success() {
            return Err(format!("Git clone failed with status: {status}"));
        }
    }

    if spec.id == inference::SENSE_VOICE_MODEL_ID {
        window
            .emit("download-log", "正在准备 Silero VAD 模型...")
            .unwrap_or(());
        ensure_silero_vad_model(&app, state.inner()).await?;
    } else {
        window
            .emit("download-log", "正在准备 Silero VAD 和中英文标点模型...")
            .unwrap_or(());
        ensure_silero_vad_model(&app, state.inner()).await?;
        ensure_punctuation_model(&app, state.inner()).await?;
    }
    window.emit("download-progress", "100%").unwrap_or(());

    eprintln!("========================================");
    eprintln!("Model downloaded successfully!");
    eprintln!("Model directory: {}", model_dir.display());
    eprintln!("========================================");
    debug_log!("Model download directory: {}", model_dir.display());
    Ok(())
}

#[command]
async fn process_audio_chunk(
    state: State<'_, AppState>,
    chunk: Vec<f32>,
) -> Result<inference::LocalTranscriptResult, String> {
    let mut model_guard = state.model.lock().await;
    if let Some(model) = model_guard.as_mut() {
        model.process(&chunk).map_err(|e| e.to_string())
    } else {
        Err("模型未加载。请在设置中下载模型后再使用本地推理。".to_string())
    }
}

#[command]
async fn flush_local_model(
    state: State<'_, AppState>,
) -> Result<inference::LocalTranscriptResult, String> {
    let mut model_guard = state.model.lock().await;
    if let Some(model) = model_guard.as_mut() {
        model.flush().map_err(|error| error.to_string())
    } else {
        Ok(inference::LocalTranscriptResult::default())
    }
}

#[command]
fn test_logging() -> Result<String, String> {
    let env_val = std::env::var("TAURI_LLM_DEBUG").unwrap_or_else(|_| String::from("not_set"));
    debug_log!(
        "Test logging called! Environment variable TAURI_LLM_DEBUG = {}",
        env_val
    );

    // Also return the value so frontend can see it
    Ok(format!("TAURI_LLM_DEBUG = {}", env_val))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Print diagnostic info on startup
    let env_val = std::env::var("TAURI_LLM_DEBUG").unwrap_or_else(|_| String::from("not_set"));
    eprintln!("=== Tauri LLM Application Starting ===");
    eprintln!("TAURI_LLM_DEBUG environment variable: {}", env_val);
    eprintln!(
        "Debug logging enabled: {}",
        env_val == "1" || env_val.to_lowercase() == "true"
    );
    eprintln!("======================================");

    debug_log!("Application initialization started");

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .manage(AppState::new())
        .manage(AudioCaptureState::new())
        .manage(AssistantState::default())
        .manage(RecordingStoreState::default())
        .manage(SpeakerDiarizationState::new())
        .setup(|app| {
            debug_log!("Setup hook called");
            let handle = app.handle().clone();

            // Register initial shortcuts on startup
            tauri::async_runtime::spawn(async move {
                let settings = load_settings(handle.clone()).unwrap_or_default();
                if let Err(error) = register_shortcuts(&handle, &settings) {
                    eprintln!("Failed to register shortcuts on startup: {error}");
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            save_settings,
            load_settings,
            download_model,
            prepare_local_model,
            process_audio_chunk,
            flush_local_model,
            test_logging,
            check_model_downloaded,
            list_visual_windows,
            capture_visual_window,
            chat_with_assistant,
            list_provider_models,
            translate_caption,
            list_audio_processes,
            audio_capture::start_process_audio_capture,
            audio_capture::start_process_audio_transcription,
            audio_capture::stop_process_audio_transcription,
            audio_capture::is_process_audio_capture_supported,
            recording_store::begin_recording,
            recording_store::append_recording_audio,
            recording_store::cancel_recording,
            recording_store::finish_recording,
            recording_store::list_recordings,
            recording_store::diarize_recording,
            recording_store::generate_recording_report,
            recording_store::export_recording
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
