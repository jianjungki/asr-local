use crate::assistant_orchestrator::{AssistantModelSet, AssistantRequest, AssistantState};
use crate::model_runtime::{ConversationMessage, ModelTarget};
use crate::speaker_diarization::{self, SpeakerSegment};
use chrono::{DateTime, Local, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};

const RECORD_FILE: &str = "record.json";
const AUDIO_FILE: &str = "audio.wav";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingEntry {
    pub id: String,
    pub title: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub duration_ms: u64,
    pub audio_source: String,
    pub audio_path: String,
    pub raw_transcript: String,
    pub formatted_transcript: Option<String>,
    pub formatting_status: String,
    pub formatting_error: Option<String>,
    pub report: Option<String>,
    pub report_status: String,
    pub report_error: Option<String>,
    pub text_provider: Option<String>,
    pub text_model: Option<String>,
    #[serde(default)]
    pub speaker_count: u32,
    #[serde(default)]
    pub speaker_segments: Vec<SpeakerSegment>,
    #[serde(default = "default_diarization_status")]
    pub diarization_status: String,
    #[serde(default)]
    pub diarization_error: Option<String>,
}

fn default_diarization_status() -> String {
    "none".to_string()
}

struct ActiveRecording {
    entry: RecordingEntry,
    directory: PathBuf,
    writer: Option<hound::WavWriter<BufWriter<fs::File>>>,
}

#[derive(Clone, Default)]
pub struct RecordingStoreState {
    active: Arc<Mutex<Option<ActiveRecording>>>,
}

impl RecordingStoreState {
    pub fn append_samples(&self, samples: &[f32]) -> Result<(), String> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| "录音存储状态不可用".to_string())?;
        let Some(session) = active.as_mut() else {
            return Ok(());
        };
        let Some(writer) = session.writer.as_mut() else {
            return Ok(());
        };

        for sample in samples {
            let normalized = sample.clamp(-1.0, 1.0);
            let value = if normalized < 0.0 {
                normalized * 32768.0
            } else {
                normalized * 32767.0
            } as i16;
            writer
                .write_sample(value)
                .map_err(|error| format!("写入录音失败：{error}"))?;
        }
        Ok(())
    }
}

fn recordings_root(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map(|path| path.join("recordings"))
        .map_err(|error| error.to_string())
}

fn record_path(directory: &Path) -> PathBuf {
    directory.join(RECORD_FILE)
}

fn save_record(directory: &Path, record: &RecordingEntry) -> Result<(), String> {
    fs::create_dir_all(directory).map_err(|error| format!("创建录音目录失败：{error}"))?;
    let json = serde_json::to_vec_pretty(record).map_err(|error| error.to_string())?;
    fs::write(record_path(directory), json).map_err(|error| format!("保存录音记录失败：{error}"))
}

fn load_record(app: &AppHandle, record_id: &str) -> Result<(PathBuf, RecordingEntry), String> {
    validate_record_id(record_id)?;
    let directory = recordings_root(app)?.join(record_id);
    let bytes =
        fs::read(record_path(&directory)).map_err(|error| format!("读取录音记录失败：{error}"))?;
    let record =
        serde_json::from_slice(&bytes).map_err(|error| format!("解析录音记录失败：{error}"))?;
    Ok((directory, record))
}

fn validate_record_id(record_id: &str) -> Result<(), String> {
    if record_id.is_empty()
        || !record_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err("录音记录编号无效".to_string());
    }
    Ok(())
}

fn recording_title(transcript: &str, started_at: DateTime<Utc>) -> String {
    let first_line = transcript
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default();
    let title = first_line.chars().take(28).collect::<String>();
    if title.is_empty() {
        format!(
            "录音 {}",
            started_at.with_timezone(&Local).format("%m-%d %H:%M")
        )
    } else if first_line.chars().count() > 28 {
        format!("{title}...")
    } else {
        title
    }
}

fn emit_recording_updated(app: &AppHandle, record: &RecordingEntry) {
    let _ = app.emit("recording-updated", record.clone());
}

#[tauri::command]
pub fn begin_recording(
    app: AppHandle,
    state: State<'_, RecordingStoreState>,
    audio_source: String,
    diarize: bool,
) -> Result<String, String> {
    let mut active = state
        .active
        .lock()
        .map_err(|_| "录音存储状态不可用".to_string())?;
    if active.is_some() {
        return Err("已有录音正在保存".to_string());
    }

    let now = Utc::now();
    let id = now.format("%Y%m%d-%H%M%S-%3f").to_string();
    let directory = recordings_root(&app)?.join(&id);
    fs::create_dir_all(&directory).map_err(|error| format!("创建录音目录失败：{error}"))?;
    let audio_path = directory.join(AUDIO_FILE);
    let writer = hound::WavWriter::create(
        &audio_path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .map_err(|error| format!("创建录音文件失败：{error}"))?;

    let entry = RecordingEntry {
        id: id.clone(),
        title: recording_title("", now),
        started_at: now.to_rfc3339_opts(SecondsFormat::Millis, true),
        ended_at: None,
        duration_ms: 0,
        audio_source,
        audio_path: audio_path.to_string_lossy().into_owned(),
        raw_transcript: String::new(),
        formatted_transcript: None,
        formatting_status: "recording".to_string(),
        formatting_error: None,
        report: None,
        report_status: "none".to_string(),
        report_error: None,
        text_provider: None,
        text_model: None,
        speaker_count: 0,
        speaker_segments: Vec::new(),
        diarization_status: "none".to_string(),
        diarization_error: None,
    };
    save_record(&directory, &entry)?;

    *active = Some(ActiveRecording {
        entry,
        directory,
        writer: Some(writer),
    });
    speaker_diarization::start_realtime(&app, diarize);
    Ok(id)
}

#[tauri::command]
pub fn append_recording_audio(
    app: AppHandle,
    state: State<'_, RecordingStoreState>,
    samples: Vec<f32>,
) -> Result<(), String> {
    state.append_samples(&samples)?;
    speaker_diarization::append_realtime_samples(&app, &samples);
    Ok(())
}

#[tauri::command]
pub fn cancel_recording(
    app: AppHandle,
    state: State<'_, RecordingStoreState>,
) -> Result<(), String> {
    speaker_diarization::stop_realtime(&app);
    let mut active = state
        .active
        .lock()
        .map_err(|_| "录音存储状态不可用".to_string())?;
    if let Some(mut session) = active.take() {
        if let Some(writer) = session.writer.take() {
            let _ = writer.finalize();
        }
        let _ = fs::remove_dir_all(session.directory);
    }
    Ok(())
}

#[tauri::command]
pub fn finish_recording(
    app: AppHandle,
    state: State<'_, RecordingStoreState>,
    transcript: String,
    auto_format: bool,
    text_model: Option<ModelTarget>,
    diarize: bool,
) -> Result<RecordingEntry, String> {
    speaker_diarization::stop_realtime(&app);
    let mut active_guard = state
        .active
        .lock()
        .map_err(|_| "录音存储状态不可用".to_string())?;
    let mut session = active_guard
        .take()
        .ok_or_else(|| "没有正在保存的录音".to_string())?;
    drop(active_guard);

    if let Some(writer) = session.writer.take() {
        writer
            .finalize()
            .map_err(|error| format!("完成录音文件失败：{error}"))?;
    }
    let ended_at = Utc::now();
    let started_at = DateTime::parse_from_rfc3339(&session.entry.started_at)
        .map(|value| value.with_timezone(&Utc))
        .unwrap_or(ended_at);
    let transcript = transcript.trim().to_string();
    session.entry.title = recording_title(&transcript, started_at);
    session.entry.ended_at = Some(ended_at.to_rfc3339_opts(SecondsFormat::Millis, true));
    session.entry.duration_ms = ended_at
        .signed_duration_since(started_at)
        .num_milliseconds()
        .max(0) as u64;
    session.entry.raw_transcript = transcript.clone();
    session.entry.formatting_status =
        if auto_format && text_model.is_some() && !transcript.is_empty() {
            "pending"
        } else {
            "none"
        }
        .to_string();
    if let Some(target) = text_model.as_ref() {
        session.entry.text_provider = Some(target.provider.clone());
        session.entry.text_model = Some(target.model.clone());
    }
    session.entry.diarization_status = if diarize { "pending" } else { "none" }.to_string();
    session.entry.diarization_error = None;
    save_record(&session.directory, &session.entry)?;
    emit_recording_updated(&app, &session.entry);

    if auto_format && !transcript.is_empty() {
        if let Some(target) = text_model {
            let app_for_task = app.clone();
            let record_id = session.entry.id.clone();
            tauri::async_runtime::spawn(async move {
                let result = {
                    let assistant = app_for_task.state::<AssistantState>();
                    assistant
                        .respond(AssistantRequest {
                            models: AssistantModelSet {
                                controller: target,
                                vision: None,
                                speech: None,
                            },
                            system_prompt: Some(
                                "你是录音文字整理助手。只修正标点、明显错别字和段落结构，保留原意、事实、语气和信息完整性；不要总结，不要新增内容。只返回整理后的正文。"
                                    .to_string(),
                            ),
                            messages: vec![ConversationMessage {
                                role: "user".to_string(),
                                content: transcript,
                            }],
                            image_data_url: None,
                            allow_desktop_actions: false,
                        })
                        .await
                };
                let update_result = (|| -> Result<(), String> {
                    let (directory, mut record) = load_record(&app_for_task, &record_id)?;
                    match result {
                        Ok(formatted) => {
                            record.formatted_transcript = Some(formatted);
                            record.formatting_status = "ready".to_string();
                            record.formatting_error = None;
                        }
                        Err(error) => {
                            record.formatting_status = "failed".to_string();
                            record.formatting_error = Some(error);
                        }
                    }
                    save_record(&directory, &record)?;
                    emit_recording_updated(&app_for_task, &record);
                    Ok(())
                })();
                if let Err(error) = update_result {
                    eprintln!("Failed to update formatted recording: {error}");
                }
            });
        }
    }

    if diarize {
        let app_for_task = app.clone();
        let record_id = session.entry.id.clone();
        tauri::async_runtime::spawn(async move {
            if let Err(error) = run_speaker_diarization(app_for_task, record_id).await {
                eprintln!("Failed to diarize recording: {error}");
            }
        });
    }

    Ok(session.entry)
}

async fn run_speaker_diarization(
    app: AppHandle,
    record_id: String,
) -> Result<RecordingEntry, String> {
    let (_, record) = load_record(&app, &record_id)?;
    let audio_path = PathBuf::from(record.audio_path);
    let result = speaker_diarization::diarize_recording_audio(&app, audio_path).await;

    let (directory, mut latest) = load_record(&app, &record_id)?;
    match result {
        Ok(output) => {
            latest.speaker_count = output.speaker_count;
            latest.speaker_segments = output.segments;
            latest.diarization_status = "ready".to_string();
            latest.diarization_error = None;
            save_record(&directory, &latest)?;
            emit_recording_updated(&app, &latest);
            Ok(latest)
        }
        Err(error) => {
            latest.diarization_status = "failed".to_string();
            latest.diarization_error = Some(error.clone());
            save_record(&directory, &latest)?;
            emit_recording_updated(&app, &latest);
            Err(error)
        }
    }
}

#[tauri::command]
pub async fn diarize_recording(
    app: AppHandle,
    record_id: String,
) -> Result<RecordingEntry, String> {
    let (directory, mut record) = load_record(&app, &record_id)?;
    record.diarization_status = "pending".to_string();
    record.diarization_error = None;
    record.speaker_count = 0;
    record.speaker_segments.clear();
    save_record(&directory, &record)?;
    emit_recording_updated(&app, &record);
    run_speaker_diarization(app, record_id).await
}

#[tauri::command]
pub fn list_recordings(app: AppHandle) -> Result<Vec<RecordingEntry>, String> {
    let root = recordings_root(&app)?;
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut records = Vec::new();
    for item in fs::read_dir(root).map_err(|error| format!("读取录音目录失败：{error}"))? {
        let Ok(item) = item else { continue };
        if !item.path().is_dir() {
            continue;
        }
        let Ok(bytes) = fs::read(record_path(&item.path())) else {
            continue;
        };
        if let Ok(record) = serde_json::from_slice::<RecordingEntry>(&bytes) {
            records.push(record);
        }
    }
    records.sort_unstable_by(|left, right| right.started_at.cmp(&left.started_at));
    Ok(records)
}

#[tauri::command]
pub async fn generate_recording_report(
    app: AppHandle,
    assistant: State<'_, AssistantState>,
    record_id: String,
    text_model: ModelTarget,
) -> Result<RecordingEntry, String> {
    let (directory, mut record) = load_record(&app, &record_id)?;
    let transcript = record
        .formatted_transcript
        .as_deref()
        .filter(|text| !text.trim().is_empty())
        .unwrap_or(&record.raw_transcript)
        .trim()
        .to_string();
    if transcript.is_empty() {
        return Err("这条录音没有可汇总的文字".to_string());
    }

    record.report_status = "generating".to_string();
    record.report_error = None;
    record.text_provider = Some(text_model.provider.clone());
    record.text_model = Some(text_model.model.clone());
    save_record(&directory, &record)?;
    emit_recording_updated(&app, &record);

    let result = assistant
        .respond(AssistantRequest {
            models: AssistantModelSet {
                controller: text_model,
                vision: None,
                speech: None,
            },
            system_prompt: Some(
                "你是录音报告整理助手。严格依据录音文字生成中文 Markdown 报告。包含简要概述、重点信息、决定与待办（没有则明确写无）以及详细记录。不得补充录音中没有的事实。"
                    .to_string(),
            ),
            messages: vec![ConversationMessage {
                role: "user".to_string(),
                content: transcript,
            }],
            image_data_url: None,
            allow_desktop_actions: false,
        })
        .await;

    let (directory, mut record) = load_record(&app, &record_id)?;
    match result {
        Ok(report) => {
            record.report = Some(report);
            record.report_status = "ready".to_string();
            record.report_error = None;
            save_record(&directory, &record)?;
            emit_recording_updated(&app, &record);
            Ok(record)
        }
        Err(error) => {
            record.report_status = "failed".to_string();
            record.report_error = Some(error.clone());
            save_record(&directory, &record)?;
            emit_recording_updated(&app, &record);
            Err(error)
        }
    }
}

fn markdown_document(record: &RecordingEntry) -> String {
    let transcript = record
        .formatted_transcript
        .as_deref()
        .filter(|text| !text.trim().is_empty())
        .unwrap_or(&record.raw_transcript);
    let ended_at = record.ended_at.as_deref().unwrap_or(&record.started_at);
    let source = match record.audio_source.as_str() {
        "process" => "程序音频",
        "microphone_and_process" => "麦克风 + 程序音频",
        _ => "麦克风",
    };
    let mut content = format!(
        "# {}\n\n- 时间：{}\n- 时长：{}\n- 来源：{}\n\n## 文字记录\n\n{}\n",
        record.title,
        ended_at,
        format_duration(record.duration_ms),
        source,
        transcript.trim()
    );
    if !record.speaker_segments.is_empty() {
        content.push_str(&format!(
            "\n## 说话人分离\n\n共检测到 {} 位说话人。\n\n",
            record.speaker_count
        ));
        for segment in &record.speaker_segments {
            content.push_str(&format!(
                "- {} - {}：说话人 {}\n",
                format_offset(segment.start_ms),
                format_offset(segment.end_ms),
                segment.speaker + 1
            ));
        }
    }
    if let Some(report) = record
        .report
        .as_deref()
        .filter(|report| !report.trim().is_empty())
    {
        content.push_str("\n## 汇总报告\n\n");
        content.push_str(report.trim());
        content.push('\n');
    }
    content
}

fn format_duration(duration_ms: u64) -> String {
    let total_seconds = duration_ms / 1_000;
    format!("{:02}:{:02}", total_seconds / 60, total_seconds % 60)
}

fn format_offset(offset_ms: u64) -> String {
    let total_seconds = offset_ms / 1_000;
    format!(
        "{:02}:{:02}.{:03}",
        total_seconds / 60,
        total_seconds % 60,
        offset_ms % 1_000
    )
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn html_document(record: &RecordingEntry) -> String {
    let transcript = record
        .formatted_transcript
        .as_deref()
        .filter(|text| !text.trim().is_empty())
        .unwrap_or(&record.raw_transcript);
    let report = record
        .report
        .as_deref()
        .filter(|text| !text.trim().is_empty())
        .map(|text| {
            format!(
                "<section><h2>汇总报告</h2><div class=\"content\">{}</div></section>",
                escape_html(text)
            )
        })
        .unwrap_or_default();
    let speaker_timeline = if record.speaker_segments.is_empty() {
        String::new()
    } else {
        let rows = record
            .speaker_segments
            .iter()
            .map(|segment| {
                format!(
                    "<li><span>{} - {}</span><strong>说话人 {}</strong></li>",
                    format_offset(segment.start_ms),
                    format_offset(segment.end_ms),
                    segment.speaker + 1
                )
            })
            .collect::<String>();
        format!(
            "<section><h2>说话人分离</h2><p>共检测到 {} 位说话人</p><ul class=\"speakers\">{rows}</ul></section>",
            record.speaker_count
        )
    };
    format!(
        r#"<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 18mm; }} body {{ color:#17191f; font-family:"Microsoft YaHei","SimHei",sans-serif; font-size:11pt; line-height:1.75; }}
h1 {{ font-size:22pt; margin:0 0 8mm; }} h2 {{ font-size:15pt; margin:9mm 0 3mm; border-bottom:1px solid #d9dde5; padding-bottom:2mm; }}
 .meta {{ color:#606775; margin-bottom:8mm; }} .content {{ white-space:pre-wrap; overflow-wrap:anywhere; }}
 .speakers {{ list-style:none; padding:0; }} .speakers li {{ display:flex; justify-content:space-between; border-bottom:1px solid #eceef2; padding:2mm 0; }}
 </style></head><body><h1>{}</h1><div class="meta">{} · {}</div><section><h2>文字记录</h2><div class="content">{}</div></section>{}{}</body></html>"#,
        escape_html(&record.title),
        escape_html(record.ended_at.as_deref().unwrap_or(&record.started_at)),
        format_duration(record.duration_ms),
        escape_html(transcript),
        speaker_timeline,
        report
    )
}

#[cfg(windows)]
fn edge_executable() -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("PROGRAMFILES(X86)")
            .map(PathBuf::from)
            .map(|path| path.join("Microsoft/Edge/Application/msedge.exe")),
        std::env::var_os("PROGRAMFILES")
            .map(PathBuf::from)
            .map(|path| path.join("Microsoft/Edge/Application/msedge.exe")),
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .map(|path| path.join("Microsoft/Edge/Application/msedge.exe")),
    ];
    candidates.into_iter().flatten().find(|path| path.exists())
}

#[cfg(windows)]
fn create_pdf(directory: &Path, record: &RecordingEntry) -> Result<PathBuf, String> {
    let edge = edge_executable().ok_or_else(|| "未找到可用于生成 PDF 的浏览器".to_string())?;
    let html_path = directory.join("recording-print.html");
    let pdf_path = directory.join("recording.pdf");
    fs::write(&html_path, html_document(record))
        .map_err(|error| format!("准备 PDF 失败：{error}"))?;
    let file_url = format!(
        "file:///{}",
        html_path
            .to_string_lossy()
            .replace('\\', "/")
            .replace(' ', "%20")
    );
    let output = Command::new(edge)
        .arg("--headless")
        .arg("--disable-gpu")
        .arg("--no-pdf-header-footer")
        .arg(format!("--print-to-pdf={}", pdf_path.to_string_lossy()))
        .arg(file_url)
        .output()
        .map_err(|error| format!("启动 PDF 生成失败：{error}"))?;
    let _ = fs::remove_file(&html_path);
    if !output.status.success() || !pdf_path.exists() {
        return Err(format!(
            "生成 PDF 失败：{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(pdf_path)
}

#[cfg(not(windows))]
fn create_pdf(_directory: &Path, _record: &RecordingEntry) -> Result<PathBuf, String> {
    Err("当前系统暂不支持直接生成 PDF".to_string())
}

#[tauri::command]
pub fn export_recording(
    app: AppHandle,
    record_id: String,
    format: String,
) -> Result<String, String> {
    let (directory, record) = load_record(&app, &record_id)?;
    let path = match format.as_str() {
        "markdown" => {
            let path = directory.join("recording.md");
            fs::write(&path, markdown_document(&record))
                .map_err(|error| format!("导出 Markdown 失败：{error}"))?;
            path
        }
        "pdf" => create_pdf(&directory, &record)?,
        _ => return Err("不支持的导出格式".to_string()),
    };
    Ok(path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_is_limited_by_characters() {
        let started_at = Utc::now();
        let title = recording_title(
            "这是一个很长很长的录音文字标题，需要在列表里保持紧凑并且不能截断中文字符",
            started_at,
        );
        assert!(title.ends_with("..."));
        assert_eq!(title.trim_end_matches("...").chars().count(), 28);
    }

    #[test]
    fn html_content_is_escaped() {
        assert_eq!(escape_html("<script>\"&"), "&lt;script&gt;&quot;&amp;");
    }

    #[test]
    fn markdown_contains_transcript_and_report() {
        let mut record = sample_record();
        record.speaker_count = 2;
        record.speaker_segments = vec![SpeakerSegment {
            start_ms: 1_250,
            end_ms: 3_500,
            speaker: 0,
        }];
        let markdown = markdown_document(&record);
        assert!(markdown.contains("## 文字记录"));
        assert!(markdown.contains("整理后的正文"));
        assert!(markdown.contains("## 汇总报告"));
        assert!(markdown.contains("报告正文"));
        assert!(markdown.contains("说话人 1"));
        assert!(markdown.contains("00:01.250 - 00:03.500"));
    }

    #[test]
    fn old_records_default_speaker_diarization_fields() {
        let mut value = serde_json::to_value(sample_record()).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("speakerCount");
        object.remove("speakerSegments");
        object.remove("diarizationStatus");
        object.remove("diarizationError");

        let record: RecordingEntry = serde_json::from_value(value).unwrap();
        assert_eq!(record.speaker_count, 0);
        assert!(record.speaker_segments.is_empty());
        assert_eq!(record.diarization_status, "none");
        assert!(record.diarization_error.is_none());
    }

    #[test]
    fn markdown_labels_combined_audio_source() {
        let mut record = sample_record();
        record.audio_source = "microphone_and_process".to_string();

        assert!(markdown_document(&record).contains("- 来源：麦克风 + 程序音频"));
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "Launches the locally installed browser to verify PDF output."]
    fn creates_a_pdf_file() {
        let directory = std::env::temp_dir().join(format!(
            "tauri-llm-pdf-test-{}",
            Utc::now().timestamp_millis()
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = create_pdf(&directory, &sample_record()).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"%PDF-"));
        let _ = fs::remove_dir_all(directory);
    }

    fn sample_record() -> RecordingEntry {
        RecordingEntry {
            id: "test-record".to_string(),
            title: "测试录音".to_string(),
            started_at: "2026-08-05T08:00:00.000Z".to_string(),
            ended_at: Some("2026-08-05T08:01:00.000Z".to_string()),
            duration_ms: 60_000,
            audio_source: "microphone".to_string(),
            audio_path: "audio.wav".to_string(),
            raw_transcript: "原始正文".to_string(),
            formatted_transcript: Some("整理后的正文".to_string()),
            formatting_status: "ready".to_string(),
            formatting_error: None,
            report: Some("报告正文".to_string()),
            report_status: "ready".to_string(),
            report_error: None,
            text_provider: Some("dashscope".to_string()),
            text_model: Some("qwen-plus".to_string()),
            speaker_count: 0,
            speaker_segments: Vec::new(),
            diarization_status: "none".to_string(),
            diarization_error: None,
        }
    }
}
