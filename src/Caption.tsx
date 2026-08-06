import { For, Show, createEffect, createMemo, createSignal, onCleanup, onMount, type JSX } from "solid-js";
import { emit, listen, type UnlistenFn } from "@tauri-apps/api/event";
import { Effect, PhysicalPosition, Window, currentMonitor, primaryMonitor } from "@tauri-apps/api/window";
import { invoke } from "@tauri-apps/api/core";
import { EyeOff, Mic2, Pause, Pin, Play, Settings2, X } from "lucide-solid";
import { useSpeechRecognition } from "./hooks/useSpeechRecognition";
import type { SenseVoiceLanguage } from "./lib/languages";
import {
  applyCaptionResult,
  captionSegmentsToText,
  fitCaptionFontSize,
  layoutCaptionLineRuns,
  layoutCaptionLines,
  type TimedCaptionSegment,
} from "./lib/captionTimeline";

type ProviderId = "dashscope" | "openai" | "anthropic" | "gemini" | "deepseek" | "siliconflow" | "openrouter" | "ollama" | "custom";

interface ProviderConfigPayload {
  apiKey?: string;
  baseUrl?: string;
}

interface SettingsPayload {
  text: string;
  opacity: number;
  fontSize: number;
  fontColor: string;
  backgroundColor: string;
  alwaysOnTop: boolean;
  captionPosition: "top" | "bottom" | "free";
  captionBackgroundStyle: "glass" | "solid" | "outline";
  modelType: "online" | "local";
  localSpeechModel: "paraformer-streaming" | "sense-voice" | "zipformer-zh";
  senseVoiceLanguage: SenseVoiceLanguage;
  audioSource: "microphone" | "process" | "microphone_and_process";
  audioInputDeviceId: string;
  processAudioPid: number;
  speechProvider?: ProviderId;
  assistantTextProvider?: ProviderId;
  assistantTextModel?: string;
  autoFormatTranscripts?: boolean;
  speakerDiarizationEnabled?: boolean;
  providerConfigs?: Partial<Record<ProviderId, ProviderConfigPayload>>;
}

interface ModelTargetPayload {
  provider: ProviderId;
  adapter: string;
  baseUrl: string;
  apiKey: string | null;
  model: string;
}

interface RealtimeSpeakerEvent {
  speaker: number;
  speakerCount: number;
  confidence: number;
  analyzedEndMs: number;
}

interface RealtimeSpeakerStatusEvent {
  status: "idle" | "loading" | "ready" | "failed";
  error: string | null;
}

type ContextMenuState = { open: boolean; x: number; y: number };
type CaptionPosition = SettingsPayload["captionPosition"];
type CaptionBackgroundStyle = SettingsPayload["captionBackgroundStyle"];

function hexToRgba(hexColor: string, opacity: number) {
  const hex = hexColor.replace("#", "");
  const value = Number.parseInt(hex, 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${opacity})`;
}

function formatDuration(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, "0");
  const seconds = Math.floor(totalSeconds % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

const providerAdapters: Record<ProviderId, string> = {
  dashscope: "aliyun",
  openai: "openai",
  anthropic: "anthropic",
  gemini: "gemini",
  deepseek: "deepseek",
  siliconflow: "openai",
  openrouter: "open_router",
  ollama: "ollama",
  custom: "openai",
};

export default function Caption() {
  const [standbyText, setStandbyText] = createSignal("字幕会显示在这里");
  const [fontSize, setFontSize] = createSignal(24);
  const [fontColor, setFontColor] = createSignal("#FFFFFF");
  const [background, setBackground] = createSignal("rgba(17, 19, 24, 0.88)");
  const [captionBackgroundStyle, setCaptionBackgroundStyle] = createSignal<CaptionBackgroundStyle>("glass");
  const [modelType, setModelType] = createSignal<"online" | "local">("online");
  const [localSpeechModel, setLocalSpeechModel] = createSignal<"paraformer-streaming" | "sense-voice" | "zipformer-zh">("paraformer-streaming");
  const [senseVoiceLanguage, setSenseVoiceLanguage] = createSignal<SenseVoiceLanguage>("auto");
  const [audioSource, setAudioSource] = createSignal<"microphone" | "process" | "microphone_and_process">("microphone");
  const [audioInputDeviceId, setAudioInputDeviceId] = createSignal("");
  const [processAudioPid, setProcessAudioPid] = createSignal(0);
  const [onlineApiKey, setOnlineApiKey] = createSignal("");
  const [autoFormatTranscripts, setAutoFormatTranscripts] = createSignal(false);
  const [speakerDiarizationEnabled, setSpeakerDiarizationEnabled] = createSignal(true);
  const [textModelTarget, setTextModelTarget] = createSignal<ModelTargetPayload | null>(null);
  const [pinned, setPinned] = createSignal(true);
  const [elapsedSeconds, setElapsedSeconds] = createSignal(0);
  const [contextMenu, setContextMenu] = createSignal<ContextMenuState>({ open: false, x: 0, y: 0 });

  const {
    isRecording,
    currentCaption,
    captionResult,
    error,
    startRecording,
    stopRecording,
  } = useSpeechRecognition();

  let recordingStartedAt = 0;
  let timer: number | undefined;
  let unlistenSettings: UnlistenFn | undefined;
  let unlistenToggle: UnlistenFn | undefined;
  let unlistenStart: UnlistenFn | undefined;
  let unlistenStop: UnlistenFn | undefined;
  let unlistenRealtimeSpeaker: UnlistenFn | undefined;
  let unlistenRealtimeSpeakerStatus: UnlistenFn | undefined;
  let captionCopyElement: HTMLDivElement | undefined;
  let captionResizeObserver: ResizeObserver | undefined;
  let previousRecordingState = false;
  let recordingSessionActive = false;
  let finalizingRecording = false;

  const [captionSegments, setCaptionSegments] = createSignal<TimedCaptionSegment[]>([]);
  const [captionCopyWidth, setCaptionCopyWidth] = createSignal(280);
  const [hasSessionResult, setHasSessionResult] = createSignal(false);
  const [activeSpeaker, setActiveSpeaker] = createSignal<number | null>(null);
  const [realtimeSpeakerCount, setRealtimeSpeakerCount] = createSignal(0);

  createEffect(() => {
    const recording = isRecording();
    if (recording && !previousRecordingState) {
      setCaptionSegments([]);
      setHasSessionResult(false);
      setActiveSpeaker(null);
      setRealtimeSpeakerCount(0);
    } else if (!recording && previousRecordingState && recordingSessionActive && !finalizingRecording) {
      void finalizeRecording();
    }
    previousRecordingState = recording;
  });

  createEffect(() => {
    const result = captionResult();
    if (!result) return;
    setHasSessionResult(true);
    setCaptionSegments((segments) => applyCaptionResult(
      segments,
      result,
      activeSpeaker() ?? undefined,
    ));
  });

  onMount(() => {
    captionResizeObserver = new ResizeObserver(() => {
      if (captionCopyElement) setCaptionCopyWidth(captionCopyElement.clientWidth);
    });
    if (captionCopyElement) {
      captionResizeObserver.observe(captionCopyElement);
      setCaptionCopyWidth(captionCopyElement.clientWidth);
    }
  });

  const fullText = createMemo(() => {
    const segments = captionSegments();
    if (segments.length) return segments;
    if (error()) {
      return [{ text: error()!, isFinal: true, receivedAt: Date.now() }];
    }
    if (!hasSessionResult()) {
      const waitingText = currentCaption().trim();
      if (waitingText) return [{ text: waitingText, isFinal: true, receivedAt: Date.now() }];
    }
    const standby = standbyText().trim();
    return standby ? [{ text: standby, isFinal: true, receivedAt: Date.now() }] : [];
  });

  const textLines = createMemo(() => {
    const segments = fullText();
    const preferredSize = fontSize();
    const capacity = Math.max(8, captionCopyWidth() / preferredSize);
    const lines = layoutCaptionLines(segments, capacity);
    const lineRuns = layoutCaptionLineRuns(segments, capacity);
    const fittedSize = fitCaptionFontSize(lines, preferredSize, captionCopyWidth());
    return {
      lines,
      lineRuns,
      fontSize: fittedSize,
    };
  });

  const sourceLabel = createMemo(() => {
    if (audioSource() === "microphone_and_process") {
      return `麦克风 + 程序音频 · PID ${processAudioPid()}`;
    }
    if (audioSource() === "process") {
      return `程序音频 · PID ${processAudioPid()}`;
    }
    return "麦克风输入";
  });

  const shellStyle = createMemo<JSX.CSSProperties>(() => ({
    "--caption-font-size": `${textLines().fontSize}px`,
    "--caption-color": fontColor(),
    "--caption-background": background(),
  }));

  const applyWindowEffect = async (captionWindow: Window, style: CaptionBackgroundStyle) => {
    try {
      if (style === "glass") {
        await captionWindow.setEffects({ effects: [Effect.Acrylic] });
      } else {
        await captionWindow.clearEffects();
      }
    } catch (error) {
      console.warn("Native caption window effect is unavailable; using CSS fallback:", error);
    }
  };

  const applyWindowPosition = async (captionWindow: Window, position: CaptionPosition) => {
    if (position === "free") return;

    const monitor = await currentMonitor() ?? await primaryMonitor();
    if (!monitor) return;

    const windowSize = await captionWindow.outerSize();
    const margin = Math.round(28 * monitor.scaleFactor);
    const x = monitor.workArea.position.x
      + Math.max(0, Math.round((monitor.workArea.size.width - windowSize.width) / 2));
    const y = position === "top"
      ? monitor.workArea.position.y + margin
      : monitor.workArea.position.y + monitor.workArea.size.height - windowSize.height - margin;

    await captionWindow.setPosition(new PhysicalPosition(x, y));
  };

  const applySettings = async (settings: SettingsPayload) => {
    setStandbyText(settings.text || "字幕会显示在这里");
    setFontSize(settings.fontSize || 24);
    setFontColor(settings.fontColor || "#FFFFFF");
    setBackground(hexToRgba(settings.backgroundColor || "#111318", settings.opacity ?? 0.88));
    setCaptionBackgroundStyle(settings.captionBackgroundStyle || "glass");
    setModelType(settings.modelType || "online");
    setLocalSpeechModel(settings.localSpeechModel || "paraformer-streaming");
    setSenseVoiceLanguage(settings.senseVoiceLanguage || "auto");
    setAudioSource(settings.audioSource || "microphone");
    setAudioInputDeviceId(settings.audioInputDeviceId || "");
    setProcessAudioPid(settings.processAudioPid || 0);
    const speechProvider = settings.speechProvider || "dashscope";
    setOnlineApiKey(
      settings.providerConfigs?.[speechProvider]?.apiKey?.trim()
      || (speechProvider === "dashscope" ? import.meta.env.VITE_DASHSCOPE_API_KEY || "" : ""),
    );
    setAutoFormatTranscripts(settings.autoFormatTranscripts === true);
    setSpeakerDiarizationEnabled(settings.speakerDiarizationEnabled !== false);
    const textProvider = settings.assistantTextProvider || "dashscope";
    const textConfig = settings.providerConfigs?.[textProvider];
    const textModel = settings.assistantTextModel?.trim() || "";
    setTextModelTarget(
      textConfig?.baseUrl?.trim() && textModel
        ? {
            provider: textProvider,
            adapter: providerAdapters[textProvider],
            baseUrl: textConfig.baseUrl,
            apiKey: textConfig.apiKey?.trim()
              || (textProvider === "dashscope" ? import.meta.env.VITE_DASHSCOPE_API_KEY || null : null),
            model: textModel,
          }
        : null,
    );
    setPinned(settings.alwaysOnTop);

    const captionWindow = await Window.getByLabel("caption");
    if (captionWindow) {
      await captionWindow.setAlwaysOnTop(settings.alwaysOnTop);
      await applyWindowEffect(captionWindow, settings.captionBackgroundStyle || "glass");
      await applyWindowPosition(captionWindow, settings.captionPosition || "bottom");
    }
  };

  const start = async () => {
    try {
      await invoke<string>("begin_recording", {
        audioSource: audioSource(),
        diarize: speakerDiarizationEnabled(),
      });
      recordingSessionActive = true;
      const started = await startRecording({
        modelType: modelType(),
        localSpeechModel: localSpeechModel(),
        senseVoiceLanguage: senseVoiceLanguage(),
        audioSource: audioSource(),
        audioInputDeviceId: audioInputDeviceId(),
        processAudioPid: processAudioPid(),
        onlineApiKey: onlineApiKey(),
        saveAudio: true,
      });
      if (!started) {
        await invoke("cancel_recording");
        recordingSessionActive = false;
      }
    } catch (startError) {
      recordingSessionActive = false;
      await invoke("cancel_recording").catch(() => undefined);
      console.error("Failed to start persisted recording:", startError);
    }
  };

  const finalizeRecording = async () => {
    if (!recordingSessionActive || finalizingRecording) return;
    finalizingRecording = true;
    try {
      await new Promise((resolve) => window.setTimeout(resolve, 180));
      await invoke("finish_recording", {
        transcript: captionSegmentsToText(captionSegments()),
        autoFormat: autoFormatTranscripts(),
        textModel: autoFormatTranscripts() ? textModelTarget() : null,
        diarize: speakerDiarizationEnabled(),
      });
      recordingSessionActive = false;
    } catch (finalizeError) {
      console.error("Failed to finalize recording:", finalizeError);
    } finally {
      finalizingRecording = false;
    }
  };

  const stop = async () => {
    if (!isRecording()) return;
    finalizingRecording = true;
    try {
      await stopRecording();
      await new Promise((resolve) => window.setTimeout(resolve, 180));
      if (recordingSessionActive) {
        await invoke("finish_recording", {
          transcript: captionSegmentsToText(captionSegments()),
          autoFormat: autoFormatTranscripts(),
          textModel: autoFormatTranscripts() ? textModelTarget() : null,
          diarize: speakerDiarizationEnabled(),
        });
        recordingSessionActive = false;
      }
    } catch (stopError) {
      console.error("Failed to stop persisted recording:", stopError);
    } finally {
      finalizingRecording = false;
    }
  };

  const toggleRecording = async () => {
    if (isRecording()) await stop();
    else await start();
  };

  const togglePinned = async () => {
    const next = !pinned();
    setPinned(next);
    const captionWindow = await Window.getByLabel("caption");
    await captionWindow?.setAlwaysOnTop(next);
  };

  const showSettings = async () => {
    const mainWindow = await Window.getByLabel("main");
    await mainWindow?.show();
    await mainWindow?.setFocus();
    setContextMenu({ open: false, x: 0, y: 0 });
  };

  const hideCaption = async () => {
    setContextMenu({ open: false, x: 0, y: 0 });
    const captionWindow = await Window.getByLabel("caption");
    await captionWindow?.hide();
  };

  const openContextMenu: JSX.EventHandlerUnion<HTMLDivElement, MouseEvent> = (event) => {
    event.preventDefault();
    setContextMenu({
      open: true,
      x: Math.min(event.clientX, window.innerWidth - 184),
      y: Math.min(event.clientY, window.innerHeight - 132),
    });
  };

  createEffect(() => {
    const recording = isRecording();
    void emit("recording-state", { recording });
    window.clearInterval(timer);

    if (recording) {
      recordingStartedAt = Date.now();
      setElapsedSeconds(0);
      timer = window.setInterval(() => {
        setElapsedSeconds(Math.floor((Date.now() - recordingStartedAt) / 1000));
      }, 1000);
    }
  });

  onMount(() => {
    void (async () => {
      try {
        await applySettings(await invoke<SettingsPayload>("load_settings"));
      } catch (error) {
        console.error("Failed to load caption settings:", error);
      }

      unlistenSettings = await listen<SettingsPayload>("update-settings", (event) => {
        void applySettings(event.payload);
      });
      unlistenToggle = await listen("toggle_recording_shortcut", () => {
        void toggleRecording();
      });
      unlistenStart = await listen("start_recording_shortcut", () => {
        if (!isRecording()) void start();
      });
      unlistenStop = await listen("stop_recording_shortcut", () => {
        if (isRecording()) void stop();
      });
      unlistenRealtimeSpeaker = await listen<RealtimeSpeakerEvent>("realtime-speaker", (event) => {
        setActiveSpeaker(event.payload.speaker);
        setRealtimeSpeakerCount(event.payload.speakerCount);
        setCaptionSegments((segments) => {
          const last = segments[segments.length - 1];
          if (!last || last.isFinal) return segments;
          return [
            ...segments.slice(0, -1),
            { ...last, speaker: event.payload.speaker },
          ];
        });
      });
      unlistenRealtimeSpeakerStatus = await listen<RealtimeSpeakerStatusEvent>(
        "realtime-speaker-status",
        (event) => {
          if (event.payload.status === "idle" || event.payload.status === "failed") {
            setActiveSpeaker(null);
          }
          if (event.payload.status === "failed" && event.payload.error) {
            console.warn("Realtime speaker diarization is unavailable:", event.payload.error);
          }
        },
      );
    })();
  });

  onCleanup(() => {
    window.clearInterval(timer);
    captionResizeObserver?.disconnect();
    unlistenSettings?.();
    unlistenToggle?.();
    unlistenStart?.();
    unlistenStop?.();
    unlistenRealtimeSpeaker?.();
    unlistenRealtimeSpeakerStatus?.();
    if (isRecording()) void stop();
  });

  return (
    <div
      class="caption-shell"
      classList={{
        glass: captionBackgroundStyle() === "glass",
        solid: captionBackgroundStyle() === "solid",
        outline: captionBackgroundStyle() === "outline",
      }}
      style={shellStyle()}
      onContextMenu={openContextMenu}
      onMouseDown={() => contextMenu().open && setContextMenu({ open: false, x: 0, y: 0 })}
    >
      <div class="caption-bar" data-tauri-drag-region>
        <div class="capture-status" data-tauri-drag-region>
          <span classList={{ "capture-dot": true, active: isRecording() }} />
          <span class="capture-time">{isRecording() ? formatDuration(elapsedSeconds()) : "待机"}</span>
          <span
            class={`capture-source${activeSpeaker() === null ? "" : ` caption-speaker caption-speaker-${activeSpeaker()! % 6}`}`}
          >
            {activeSpeaker() === null
              ? sourceLabel()
              : `说话人 ${activeSpeaker()! + 1}/${realtimeSpeakerCount()} · ${sourceLabel()}`}
          </span>
        </div>

        <div class="caption-copy" data-tauri-drag-region ref={captionCopyElement}>
          <Show when={textLines().lines.length > 1}>
            <p class="previous-caption">
              <For each={textLines().lineRuns[0]}>{(run) => (
                <span class={run.speaker === undefined ? undefined : `caption-speaker caption-speaker-${run.speaker % 6}`}>
                  {run.text}
                </span>
              )}</For>
            </p>
          </Show>
          <p class="current-caption">
            <Show
              when={textLines().lineRuns[textLines().lineRuns.length - 1]}
              fallback={standbyText()}
            >
              {(runs) => (
                <For each={runs()}>{(run) => (
                  <span class={run.speaker === undefined ? undefined : `caption-speaker caption-speaker-${run.speaker % 6}`}>
                    {run.text}
                  </span>
                )}</For>
              )}
            </Show>
          </p>
        </div>

        <div class="caption-actions">
          <button title={isRecording() ? "暂停录音" : "开始录音"} onClick={toggleRecording}>
            {isRecording() ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <button title={pinned() ? "取消置顶" : "始终置顶"} classList={{ active: pinned() }} onClick={togglePinned}><Pin size={17} /></button>
          <button title="隐藏字幕" onClick={hideCaption}><X size={18} /></button>
        </div>

        <span classList={{ "capture-progress": true, active: isRecording() }} />
      </div>

      <Show when={contextMenu().open}>
        <div class="caption-menu" style={{ left: `${contextMenu().x}px`, top: `${contextMenu().y}px` }} onMouseDown={(event) => event.stopPropagation()}>
          <button onClick={toggleRecording}>{isRecording() ? <Pause size={16} /> : <Mic2 size={16} />}{isRecording() ? "暂停录音" : "开始录音"}</button>
          <button onClick={showSettings}><Settings2 size={16} />打开设置</button>
          <button onClick={hideCaption}><EyeOff size={16} />隐藏字幕</button>
        </div>
      </Show>
    </div>
  );
}
