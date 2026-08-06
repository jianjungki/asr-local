import {
  For,
  Show,
  createEffect,
  createMemo,
  createSignal,
  onCleanup,
  onMount,
  type Component,
} from "solid-js";
import { createStore, unwrap } from "solid-js/store";
import { Dynamic, Portal } from "solid-js/web";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { emit, listen, type UnlistenFn } from "@tauri-apps/api/event";
import { Window } from "@tauri-apps/api/window";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { useSpeechRecognition } from "./hooks/useSpeechRecognition";
import { senseVoiceLanguages, type SenseVoiceLanguage } from "./lib/languages";
import {
  ArrowDownToLine,
  ArrowLeft,
  ArrowUpToLine,
  BoxSelect,
  BrainCircuit,
  Captions,
  Check,
  ChevronRight,
  CircleDot,
  Database,
  Download,
  FileDown,
  FileText,
  Eye,
  EyeOff,
  Cloud,
  HardDrive,
  Info,
  Keyboard,
  ListMusic,
  Layers3,
  Mic2,
  MonitorUp,
  Move,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Search,
  Server,
  Share2,
  Settings2,
  Sparkles,
  Square,
  X,
} from "lucide-solid";
import "./App.css";

type ProviderId = "dashscope" | "openai" | "anthropic" | "gemini" | "deepseek" | "siliconflow" | "openrouter" | "ollama" | "custom";
type ProviderMode = "online" | "local";

interface ProviderConfig {
  baseUrl: string;
  apiKey: string;
  textModel: string;
  visionModel: string;
  availableModels: string[];
}

interface ProviderDefinition {
  id: ProviderId;
  name: string;
  adapter: string;
  mode: ProviderMode;
  supportsVision: boolean;
  supportsSpeech: boolean;
  defaults: ProviderConfig;
}

interface Settings {
  text: string;
  alwaysOnTop: boolean;
  fontSize: number;
  fontColor: string;
  backgroundColor: string;
  opacity: number;
  captionPosition: "top" | "bottom" | "free";
  captionBackgroundStyle: "glass" | "solid" | "outline";
  modelType: "online" | "local";
  audioSource: "microphone" | "process" | "microphone_and_process";
  audioInputDeviceId: string;
  processAudioPid: number;
  localModelPath: string;
  isModelDownloaded: boolean;
  localSpeechModel: "paraformer-streaming" | "sense-voice" | "zipformer-zh";
  senseVoiceLanguage: SenseVoiceLanguage;
  speechProvider: ProviderId;
  assistantTextProvider: ProviderId;
  assistantVisionProvider: ProviderId;
  assistantTextModel: string;
  assistantVisionModel: string;
  providerConfigs: Record<ProviderId, ProviderConfig>;
  autoFormatTranscripts: boolean;
  speakerDiarizationEnabled: boolean;
  toggleRecordingHotkey: string;
  captureVisualHotkey: string;
}

type LegacySettings = Partial<Settings> & {
  assistantProvider?: "dashscope" | "ollama";
  assistantTextModel?: string;
  assistantVisionModel?: string;
  ollamaBaseUrl?: string;
  startRecordingHotkey?: string;
  stopRecordingHotkey?: string;
};

interface RecordingEntry {
  id: string;
  title: string;
  startedAt: string;
  endedAt: string | null;
  durationMs: number;
  audioSource: "microphone" | "process" | "microphone_and_process";
  audioPath: string;
  rawTranscript: string;
  formattedTranscript: string | null;
  formattingStatus: "recording" | "pending" | "ready" | "failed" | "none";
  formattingError: string | null;
  report: string | null;
  reportStatus: "none" | "generating" | "ready" | "failed";
  reportError: string | null;
  textProvider: string | null;
  textModel: string | null;
  speakerCount: number;
  speakerSegments: SpeakerSegment[];
  diarizationStatus: "none" | "pending" | "ready" | "failed";
  diarizationError: string | null;
}

interface SpeakerSegment {
  startMs: number;
  endMs: number;
  speaker: number;
}

interface AudioInputDevice {
  deviceId: string;
  label: string;
}

interface AudioProcessTarget {
  pid: number;
  name: string;
  windowTitle?: string | null;
  executablePath?: string | null;
  commandLine?: string | null;
  sessionId?: number | null;
  hasAudioSession: boolean;
  audioSessionCount: number;
}

interface VisualWindowTarget {
  id: string;
  title: string;
  processId: number;
  processName: string;
  minimized: boolean;
}

interface CapturedWindow {
  windowId: string;
  windowTitle: string;
  processName: string;
  width: number;
  height: number;
  imageDataUrl: string;
}

interface AssistantMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
}

interface AssistantModelTarget {
  provider: ProviderId;
  adapter: string;
  baseUrl: string;
  apiKey: string | null;
  model: string;
}

type SectionId = "general" | "recording" | "history" | "ai" | "providers" | "assistant" | "memory" | "captions" | "shortcuts" | "about";
type SaveState = "idle" | "saving" | "saved" | "error";

const VERSION = "0.1.0";

const providerDefinitions: ProviderDefinition[] = [
  {
    id: "dashscope",
    name: "DashScope",
    adapter: "aliyun",
    mode: "online",
    supportsVision: true,
    supportsSpeech: true,
    defaults: {
      baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1",
      apiKey: "",
      textModel: "qwen-plus",
      visionModel: "qwen-vl-plus",
      availableModels: ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-vl-plus"],
    },
  },
  {
    id: "openai",
    name: "OpenAI",
    adapter: "openai",
    mode: "online",
    supportsVision: true,
    supportsSpeech: false,
    defaults: {
      baseUrl: "https://api.openai.com/v1",
      apiKey: "",
      textModel: "gpt-4.1-mini",
      visionModel: "gpt-4.1-mini",
      availableModels: ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
    },
  },
  {
    id: "anthropic",
    name: "Anthropic",
    adapter: "anthropic",
    mode: "online",
    supportsVision: true,
    supportsSpeech: false,
    defaults: {
      baseUrl: "https://api.anthropic.com/v1",
      apiKey: "",
      textModel: "claude-sonnet-4-5",
      visionModel: "claude-sonnet-4-5",
      availableModels: ["claude-sonnet-4-5", "claude-haiku-4-5"],
    },
  },
  {
    id: "gemini",
    name: "Google Gemini",
    adapter: "gemini",
    mode: "online",
    supportsVision: true,
    supportsSpeech: false,
    defaults: {
      baseUrl: "https://generativelanguage.googleapis.com/v1beta",
      apiKey: "",
      textModel: "gemini-2.5-flash",
      visionModel: "gemini-2.5-flash",
      availableModels: ["gemini-2.5-flash", "gemini-2.5-pro"],
    },
  },
  {
    id: "deepseek",
    name: "DeepSeek",
    adapter: "deepseek",
    mode: "online",
    supportsVision: false,
    supportsSpeech: false,
    defaults: {
      baseUrl: "https://api.deepseek.com/v1",
      apiKey: "",
      textModel: "deepseek-chat",
      visionModel: "",
      availableModels: ["deepseek-chat", "deepseek-reasoner"],
    },
  },
  {
    id: "siliconflow",
    name: "硅基流动",
    adapter: "openai",
    mode: "online",
    supportsVision: true,
    supportsSpeech: false,
    defaults: {
      baseUrl: "https://api.siliconflow.cn/v1",
      apiKey: "",
      textModel: "Qwen/Qwen3-8B",
      visionModel: "Qwen/Qwen2.5-VL-7B-Instruct",
      availableModels: ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-VL-7B-Instruct"],
    },
  },
  {
    id: "openrouter",
    name: "OpenRouter",
    adapter: "open_router",
    mode: "online",
    supportsVision: true,
    supportsSpeech: false,
    defaults: {
      baseUrl: "https://openrouter.ai/api/v1",
      apiKey: "",
      textModel: "openai/gpt-4.1-mini",
      visionModel: "openai/gpt-4.1-mini",
      availableModels: ["openai/gpt-4.1-mini", "anthropic/claude-sonnet-4-5", "google/gemini-2.5-flash"],
    },
  },
  {
    id: "ollama",
    name: "Ollama",
    adapter: "ollama",
    mode: "local",
    supportsVision: true,
    supportsSpeech: false,
    defaults: {
      baseUrl: "http://127.0.0.1:11434",
      apiKey: "",
      textModel: "qwen3:4b",
      visionModel: "qwen2.5vl:7b",
      availableModels: ["qwen3:4b", "qwen2.5vl:7b"],
    },
  },
  {
    id: "custom",
    name: "自定义服务",
    adapter: "openai",
    mode: "online",
    supportsVision: true,
    supportsSpeech: false,
    defaults: {
      baseUrl: "",
      apiKey: "",
      textModel: "",
      visionModel: "",
      availableModels: [],
    },
  },
];

const createDefaultProviderConfigs = (): Record<ProviderId, ProviderConfig> =>
  Object.fromEntries(
    providerDefinitions.map((provider) => [provider.id, { ...provider.defaults }]),
  ) as Record<ProviderId, ProviderConfig>;

const mergeProviderConfigs = (
  configs?: Partial<Record<ProviderId, Partial<ProviderConfig>>>,
): Record<ProviderId, ProviderConfig> => {
  const defaults = createDefaultProviderConfigs();
  for (const provider of providerDefinitions) {
    defaults[provider.id] = {
      ...defaults[provider.id],
      ...(configs?.[provider.id] || {}),
    };
  }
  return defaults;
};

const createAssistantModelTarget = (
  provider: ProviderDefinition,
  config: ProviderConfig,
  model: string,
  fallbackApiKey = "",
): AssistantModelTarget => ({
  provider: provider.id,
  adapter: provider.adapter,
  baseUrl: config.baseUrl,
  apiKey: config.apiKey.trim() || fallbackApiKey || null,
  model: model.trim(),
});

const localSpeechModels = [
  {
    id: "paraformer-streaming" as const,
    name: "中英双语（推荐）",
    description: "适合中文、英文和中英混合内容；首次下载约 1 GB",
    directory: "sherpa-onnx-streaming-paraformer-bilingual-zh-en",
  },
  {
    id: "sense-voice" as const,
    name: "多语种",
    description: "支持中文、粤语、英语、日语和韩语",
    directory: "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
  },
  {
    id: "zipformer-zh" as const,
    name: "中文轻量版",
    description: "适合以中文为主的内容，资源占用更低",
    directory: "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03",
  },
];

const defaultSettings: Settings = {
  text: "字幕会显示在这里",
  alwaysOnTop: true,
  fontSize: 24,
  fontColor: "#FFFFFF",
  backgroundColor: "#111318",
  opacity: 0.88,
  captionPosition: "bottom",
  captionBackgroundStyle: "glass",
  modelType: "online",
  audioSource: "microphone",
  audioInputDeviceId: "",
  processAudioPid: 0,
  localModelPath: "",
  isModelDownloaded: false,
  localSpeechModel: "paraformer-streaming",
  senseVoiceLanguage: "auto",
  speechProvider: "dashscope",
  assistantTextProvider: "dashscope",
  assistantVisionProvider: "dashscope",
  assistantTextModel: "qwen-plus",
  assistantVisionModel: "qwen-vl-plus",
  providerConfigs: createDefaultProviderConfigs(),
  autoFormatTranscripts: false,
  speakerDiarizationEnabled: true,
  toggleRecordingHotkey: "Alt+F1",
  captureVisualHotkey: "Alt+F2",
};

const navItems: Array<{
  id: SectionId;
  label: string;
  description: string;
  icon: Component<{ size?: number; strokeWidth?: number }>;
}> = [
  { id: "general", label: "通用", description: "窗口与应用行为", icon: Settings2 },
  { id: "recording", label: "录音", description: "音频来源与设备", icon: Mic2 },
  { id: "history", label: "录音记录", description: "音频、文字与汇总报告", icon: ListMusic },
  { id: "ai", label: "AI 处理", description: "在线与本地识别", icon: BrainCircuit },
  { id: "providers", label: "Provider", description: "服务连接与模型", icon: Server },
  { id: "assistant", label: "AI 助手", description: "语音、文字与窗口理解", icon: Sparkles },
  { id: "memory", label: "记忆", description: "本地数据状态", icon: Database },
  { id: "captions", label: "字幕条", description: "外观与预览", icon: Captions },
  { id: "shortcuts", label: "快捷键", description: "全局录音与截图", icon: Keyboard },
  { id: "about", label: "关于", description: "版本与组件", icon: Info },
];

function Toggle(props: { checked: boolean; label: string; onChange: (checked: boolean) => void }) {
  return (
    <label class="switch-control">
      <input
        type="checkbox"
        aria-label={props.label}
        checked={props.checked}
        onChange={(event) => props.onChange(event.currentTarget.checked)}
      />
      <span class="switch-track"><span class="switch-thumb" /></span>
    </label>
  );
}

function SectionHeading(props: { eyebrow?: string; title: string; description?: string }) {
  return (
    <div class="section-heading">
      <Show when={props.eyebrow}><span class="section-eyebrow">{props.eyebrow}</span></Show>
      <h2>{props.title}</h2>
      <Show when={props.description}><p>{props.description}</p></Show>
    </div>
  );
}

function SettingRow(props: {
  title: string;
  description?: string;
  children: unknown;
  stacked?: boolean;
}) {
  return (
    <div class="setting-row" classList={{ stacked: props.stacked }}>
      <div class="setting-copy">
        <span class="setting-title">{props.title}</span>
        <Show when={props.description}><span class="setting-description">{props.description}</span></Show>
      </div>
      <div class="setting-control">{props.children as never}</div>
    </div>
  );
}

function formatRecordingTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatRecordingDuration(durationMs: number) {
  const totalSeconds = Math.max(0, Math.floor(durationMs / 1000));
  return `${Math.floor(totalSeconds / 60).toString().padStart(2, "0")}:${(totalSeconds % 60).toString().padStart(2, "0")}`;
}

function formatAudioSource(source: RecordingEntry["audioSource"]) {
  if (source === "process") return "程序音频";
  if (source === "microphone_and_process") return "麦克风 + 程序音频";
  return "麦克风";
}

function App() {
  const [activeSection, setActiveSection] = createSignal<SectionId>("recording");
  const [settings, setSettings] = createStore<Settings>(defaultSettings);
  const [audioInputDevices, setAudioInputDevices] = createSignal<AudioInputDevice[]>([]);
  const [audioProcesses, setAudioProcesses] = createSignal<AudioProcessTarget[]>([]);
  const [processSearch, setProcessSearch] = createSignal("");
  const [isProcessPickerOpen, setIsProcessPickerOpen] = createSignal(false);
  const [isDownloading, setIsDownloading] = createSignal(false);
  const [downloadProgress, setDownloadProgress] = createSignal("");
  const [isLoadingDevices, setIsLoadingDevices] = createSignal(false);
  const [isLoadingProcesses, setIsLoadingProcesses] = createSignal(false);
  const [isRecording, setIsRecording] = createSignal(false);
  const [visualWindows, setVisualWindows] = createSignal<VisualWindowTarget[]>([]);
  const [visualWindowId, setVisualWindowId] = createSignal("");
  const [capturedWindow, setCapturedWindow] = createSignal<CapturedWindow | null>(null);
  const [useVisualContext, setUseVisualContext] = createSignal(true);
  const [isLoadingVisualWindows, setIsLoadingVisualWindows] = createSignal(false);
  const [isCapturingWindow, setIsCapturingWindow] = createSignal(false);
  const [assistantInput, setAssistantInput] = createSignal("");
  const [selectedProviderId, setSelectedProviderId] = createSignal<ProviderId>("dashscope");
  const [showProviderKey, setShowProviderKey] = createSignal(false);
  const [isLoadingProviderModels, setIsLoadingProviderModels] = createSignal(false);
  const [manualProviderModel, setManualProviderModel] = createSignal("");
  const [recordings, setRecordings] = createSignal<RecordingEntry[]>([]);
  const [selectedRecordingId, setSelectedRecordingId] = createSignal("");
  const [isLoadingRecordings, setIsLoadingRecordings] = createSignal(false);
  const [isGeneratingReport, setIsGeneratingReport] = createSignal(false);
  const [isDiarizing, setIsDiarizing] = createSignal(false);
  const [exportingFormat, setExportingFormat] = createSignal<"markdown" | "pdf" | null>(null);
  const [assistantMessages, setAssistantMessages] = createSignal<AssistantMessage[]>([
    { id: 1, role: "assistant", content: "可以直接说话或输入问题；捕获窗口后，我也能结合当前页面内容回答。" },
  ]);
  const [isAssistantThinking, setIsAssistantThinking] = createSignal(false);
  const [saveState, setSaveState] = createSignal<SaveState>("idle");
  const [popupInfo, setPopupInfo] = createStore({
    show: false,
    message: "",
    type: "success" as "success" | "error",
  });

  let hydrated = false;
  let saveTimer: number | undefined;
  let unlistenRecording: UnlistenFn | undefined;
  let unlistenDownloadProgress: UnlistenFn | undefined;
  let unlistenRecordingUpdated: UnlistenFn | undefined;
  let unlistenCaptureVisual: UnlistenFn | undefined;
  let visualWindowsLoaded = false;
  let recordingsLoaded = false;
  let assistantMessageId = 1;
  let recordingAudioElement: HTMLAudioElement | undefined;

  const assistantVoice = useSpeechRecognition();

  const activeItem = createMemo(() => navItems.find((item) => item.id === activeSection())!);
  const selectedProcess = createMemo(() =>
    audioProcesses().find((process) => process.pid === settings.processAudioPid),
  );
  const selectedVisualWindow = createMemo(() =>
    visualWindows().find((window) => window.id === visualWindowId()),
  );
  const selectedLocalSpeechModel = createMemo(() =>
    localSpeechModels.find((model) => model.id === settings.localSpeechModel) || localSpeechModels[0],
  );
  const selectedProviderDefinition = createMemo(() =>
    providerDefinitions.find((provider) => provider.id === selectedProviderId()) || providerDefinitions[0],
  );
  const selectedProviderConfig = createMemo(() => settings.providerConfigs[selectedProviderId()]);
  const selectedRecording = createMemo(() =>
    recordings().find((recording) => recording.id === selectedRecordingId()) || null,
  );
  const assistantTextProvider = createMemo(() =>
    providerDefinitions.find((provider) => provider.id === settings.assistantTextProvider) || providerDefinitions[0],
  );
  const assistantVisionProvider = createMemo(() =>
    providerDefinitions.find((provider) => provider.id === settings.assistantVisionProvider) || providerDefinitions[0],
  );
  const assistantTextProviderConfig = createMemo(() => settings.providerConfigs[assistantTextProvider().id]);
  const dashscopeEnvApiKey = import.meta.env.VITE_DASHSCOPE_API_KEY || "";
  const speechProviderConfig = createMemo(() => settings.providerConfigs[settings.speechProvider]);
  const speechApiKey = createMemo(() =>
    speechProviderConfig()?.apiKey.trim() || (settings.speechProvider === "dashscope" ? dashscopeEnvApiKey : ""),
  );
  const speechProviderConfigured = createMemo(() =>
    Boolean(speechProviderConfig()?.baseUrl.trim() && speechApiKey()),
  );

  const filteredProcesses = createMemo(() => {
    const keyword = processSearch().trim().toLowerCase();
    const filtered = keyword
      ? audioProcesses().filter((process) =>
          `${process.name} ${process.pid} ${process.windowTitle || ""} ${process.executablePath || ""}`
            .toLowerCase()
            .includes(keyword),
        )
      : audioProcesses();

    return filtered
      .slice()
      .sort((a, b) =>
        Number(b.hasAudioSession) - Number(a.hasAudioSession)
        || b.audioSessionCount - a.audioSessionCount
        || a.name.localeCompare(b.name)
        || a.pid - b.pid,
      )
      .slice(0, 100);
  });

  const previewBackground = createMemo(() => {
    const hex = settings.backgroundColor.replace("#", "");
    const value = Number.parseInt(hex, 16);
    const red = (value >> 16) & 255;
    const green = (value >> 8) & 255;
    const blue = value & 255;
    const opacity = settings.captionBackgroundStyle === "glass"
      ? Math.min(0.72, Math.max(0.3, settings.opacity * 0.72))
      : settings.opacity;
    return `rgba(${red}, ${green}, ${blue}, ${opacity})`;
  });

  const showPopup = (message: string, type: "success" | "error") => {
    setPopupInfo({ show: true, message, type });
    window.setTimeout(() => setPopupInfo({ show: false, message: "", type: "success" }), 2400);
  };

  const updateSelectedProvider = (field: keyof ProviderConfig, value: string) => {
    setSettings("providerConfigs", selectedProviderId(), field, value);
  };

  const resetSelectedProvider = () => {
    const provider = selectedProviderDefinition();
    setSettings("providerConfigs", provider.id, { ...provider.defaults });
    setShowProviderKey(false);
  };

  const providerModels = (providerId: ProviderId, selectedModel = "") => {
    const config = settings.providerConfigs[providerId];
    return Array.from(new Set([
      ...(config.availableModels || []),
      config.textModel,
      config.visionModel,
      selectedModel,
    ].map((model) => model.trim()).filter(Boolean)));
  };

  const selectAssistantProvider = (kind: "text" | "vision", providerId: ProviderId) => {
    const models = providerModels(providerId);
    if (kind === "text") {
      setSettings("assistantTextProvider", providerId);
      setSettings("assistantTextModel", models[0] || "");
    } else {
      setSettings("assistantVisionProvider", providerId);
      setSettings("assistantVisionModel", models[0] || "");
    }
  };

  const refreshProviderModels = async () => {
    const provider = selectedProviderDefinition();
    const config = selectedProviderConfig();
    const apiKey = config.apiKey.trim() || (provider.id === "dashscope" ? dashscopeEnvApiKey : "");
    if (!config.baseUrl.trim()) {
      showPopup("请先填写服务地址", "error");
      return;
    }
    setIsLoadingProviderModels(true);
    try {
      const models = await invoke<string[]>("list_provider_models", {
        provider: {
          adapter: provider.adapter,
          baseUrl: config.baseUrl,
          apiKey: apiKey || null,
        },
      });
      if (!models.length) {
        showPopup("服务未返回可用模型，可手动添加", "error");
        return;
      }
      setSettings("providerConfigs", provider.id, "availableModels", models);
      if (settings.assistantTextProvider === provider.id && !models.includes(settings.assistantTextModel)) {
        setSettings("assistantTextModel", models[0]);
      }
      if (settings.assistantVisionProvider === provider.id && !models.includes(settings.assistantVisionModel)) {
        setSettings("assistantVisionModel", models[0]);
      }
      showPopup(`已获取 ${models.length} 个模型`, "success");
    } catch (fetchError) {
      console.error("Failed to list provider models:", fetchError);
      showPopup(`获取模型失败：${String(fetchError)}`, "error");
    } finally {
      setIsLoadingProviderModels(false);
    }
  };

  const addManualProviderModel = () => {
    const model = manualProviderModel().trim();
    if (!model) return;
    const providerId = selectedProviderId();
    const models = providerModels(providerId, model);
    setSettings("providerConfigs", providerId, "availableModels", models);
    setManualProviderModel("");
    showPopup("模型已添加", "success");
  };

  const loadRecordings = async () => {
    setIsLoadingRecordings(true);
    try {
      const items = await invoke<RecordingEntry[]>("list_recordings");
      setRecordings(items);
      if (selectedRecordingId() && !items.some((item) => item.id === selectedRecordingId())) {
        setSelectedRecordingId("");
      }
    } catch (loadError) {
      console.error("Failed to load recordings:", loadError);
      showPopup(`读取录音记录失败：${String(loadError)}`, "error");
    } finally {
      setIsLoadingRecordings(false);
    }
  };

  const currentTextModelTarget = () => {
    const provider = assistantTextProvider();
    const config = assistantTextProviderConfig();
    const model = settings.assistantTextModel.trim();
    const apiKey = config.apiKey.trim() || (provider.id === "dashscope" ? dashscopeEnvApiKey : "");
    if (!config.baseUrl.trim() || !model) return null;
    if (provider.mode === "online" && provider.id !== "custom" && !apiKey) return null;
    return createAssistantModelTarget(provider, config, model, provider.id === "dashscope" ? dashscopeEnvApiKey : "");
  };

  const generateRecordingReport = async () => {
    const recording = selectedRecording();
    if (!recording || isGeneratingReport()) return;
    const target = currentTextModelTarget();
    if (!target) {
      showPopup("请先选择并配置文字模型", "error");
      return;
    }
    setIsGeneratingReport(true);
    try {
      const updated = await invoke<RecordingEntry>("generate_recording_report", {
        recordId: recording.id,
        textModel: target,
      });
      setRecordings((items) => items.map((item) => item.id === updated.id ? updated : item));
      showPopup("汇总报告已生成", "success");
    } catch (reportError) {
      console.error("Failed to generate recording report:", reportError);
      showPopup(`生成报告失败：${String(reportError)}`, "error");
      await loadRecordings();
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const diarizeSelectedRecording = async () => {
    const recording = selectedRecording();
    if (!recording || isDiarizing() || recording.diarizationStatus === "pending") return;
    setIsDiarizing(true);
    try {
      const updated = await invoke<RecordingEntry>("diarize_recording", { recordId: recording.id });
      setRecordings((items) => items.map((item) => item.id === updated.id ? updated : item));
      showPopup(
        updated.speakerCount > 0 ? `检测到 ${updated.speakerCount} 位说话人` : "没有检测到清晰的发言片段",
        "success",
      );
    } catch (diarizationError) {
      console.error("Failed to diarize recording:", diarizationError);
      showPopup(`说话人分离失败：${String(diarizationError)}`, "error");
      await loadRecordings();
    } finally {
      setIsDiarizing(false);
    }
  };

  const playSpeakerSegment = (segment: SpeakerSegment) => {
    if (!recordingAudioElement) return;
    recordingAudioElement.currentTime = segment.startMs / 1_000;
    void recordingAudioElement.play();
  };

  const exportRecording = async (format: "markdown" | "pdf") => {
    const recording = selectedRecording();
    if (!recording || exportingFormat()) return;
    setExportingFormat(format);
    try {
      const path = await invoke<string>("export_recording", { recordId: recording.id, format });
      await revealItemInDir(path);
      showPopup(`${format === "pdf" ? "PDF" : "Markdown"} 已生成`, "success");
    } catch (exportError) {
      console.error("Failed to export recording:", exportError);
      showPopup(`导出失败：${String(exportError)}`, "error");
    } finally {
      setExportingFormat(null);
    }
  };

  const openProviderSettings = (providerId: ProviderId) => {
    setSelectedProviderId(providerId);
    setShowProviderKey(false);
    setActiveSection("providers");
  };

  const loadVisualWindows = async () => {
    setIsLoadingVisualWindows(true);
    try {
      const windows = await invoke<VisualWindowTarget[]>("list_visual_windows");
      setVisualWindows(windows);
      visualWindowsLoaded = true;
      if (!windows.some((window) => window.id === visualWindowId())) {
        setVisualWindowId(windows[0]?.id || "");
      }
    } catch (error) {
      console.error("Failed to list visual windows:", error);
      showPopup(`读取窗口失败：${String(error)}`, "error");
    } finally {
      setIsLoadingVisualWindows(false);
    }
  };

  const captureVisualContext = async (mode: "foreground" | "selected"): Promise<CapturedWindow | null> => {
    if (isCapturingWindow()) return null;
    if (mode === "selected" && !visualWindowId()) {
      showPopup("请先选择要理解的窗口", "error");
      return null;
    }
    if (mode === "selected" && selectedVisualWindow()?.minimized) {
      showPopup("请先恢复目标窗口，最小化窗口可能无法正确截图", "error");
      return null;
    }

    setIsCapturingWindow(true);
    const mainWindow = await Window.getByLabel("main");
    try {
      if (mode === "foreground") {
        await mainWindow?.hide();
        await new Promise((resolve) => window.setTimeout(resolve, 450));
      }
      const capture = await invoke<CapturedWindow>("capture_visual_window", {
        windowId: mode === "selected" ? visualWindowId() : null,
      });
      setCapturedWindow(capture);
      setUseVisualContext(true);
      showPopup(`已捕获：${capture.windowTitle || capture.processName}`, "success");
      return capture;
    } catch (error) {
      console.error("Failed to capture visual context:", error);
      showPopup(`窗口捕获失败：${String(error)}`, "error");
      return null;
    } finally {
      if (mode === "foreground") {
        await mainWindow?.show();
        await mainWindow?.setFocus();
      }
      setIsCapturingWindow(false);
    }
  };

  const toggleAssistantVoice = async () => {
    if (assistantVoice.isRecording()) {
      await assistantVoice.stopRecording();
      return;
    }
    if (isRecording()) {
      showPopup("请先停止实时字幕录音，再开始助手听写", "error");
      return;
    }
    setAssistantInput("");
    await assistantVoice.startRecording({
      modelType: settings.modelType,
      localSpeechModel: settings.localSpeechModel,
      senseVoiceLanguage: settings.senseVoiceLanguage,
      audioSource: "microphone",
      audioInputDeviceId: settings.audioInputDeviceId,
      onlineApiKey: speechApiKey(),
    });
  };

  const sendAssistantMessage = async (preset?: string) => {
    if (assistantVoice.isRecording()) {
      await assistantVoice.stopRecording();
      await new Promise((resolve) => window.setTimeout(resolve, 180));
    }
    const content = (preset ?? assistantInput()).trim();
    if (!content || isAssistantThinking()) return;
    const imageDataUrl = useVisualContext() ? capturedWindow()?.imageDataUrl || null : null;
    const textProvider = assistantTextProvider();
    const textProviderConfig = settings.providerConfigs[textProvider.id];
    const visionProvider = assistantVisionProvider();
    const visionProviderConfig = settings.providerConfigs[visionProvider.id];
    const provider = imageDataUrl ? visionProvider : textProvider;
    const providerConfig = settings.providerConfigs[provider.id];
    const model = (imageDataUrl ? settings.assistantVisionModel : settings.assistantTextModel).trim();
    const apiKey = providerConfig.apiKey.trim() || (provider.id === "dashscope" ? dashscopeEnvApiKey : "");
    if (!providerConfig.baseUrl.trim()) {
      showPopup(`请先在 Provider 中配置 ${provider.name} 的服务地址`, "error");
      return;
    }
    if (!model) {
      showPopup(`请先选择 ${provider.name} 的${imageDataUrl ? "视觉" : "文字"}模型`, "error");
      return;
    }
    if (provider.mode === "online" && provider.id !== "custom" && !apiKey) {
      showPopup(`请先在 Provider 中配置 ${provider.name} 的访问密钥`, "error");
      return;
    }

    const userMessage: AssistantMessage = {
      id: ++assistantMessageId,
      role: "user",
      content,
    };
    const nextMessages = [...assistantMessages(), userMessage];
    setAssistantMessages(nextMessages);
    setAssistantInput("");
    setIsAssistantThinking(true);
    try {
      const answer = await invoke<string>("chat_with_assistant", {
        models: {
          controller: createAssistantModelTarget(
            textProvider,
            textProviderConfig,
            settings.assistantTextModel,
            textProvider.id === "dashscope" ? dashscopeEnvApiKey : "",
          ),
          vision: imageDataUrl
            ? createAssistantModelTarget(
                visionProvider,
                visionProviderConfig,
                settings.assistantVisionModel,
                visionProvider.id === "dashscope" ? dashscopeEnvApiKey : "",
              )
            : null,
          speech: null,
        },
        systemPrompt: "你是桌面端语音助手。请优先用简洁、清楚的中文回答；当提供窗口截图时，要结合可见文字、界面状态和上下文，不要臆造截图中不存在的信息。",
        messages: nextMessages
          .filter((message) => message.id !== 1)
          .slice(-12)
          .map(({ role, content }) => ({ role, content })),
        imageDataUrl,
      });
      setAssistantMessages((messages) => [
        ...messages,
        { id: ++assistantMessageId, role: "assistant", content: answer },
      ]);
    } catch (error) {
      console.error("Assistant model call failed:", error);
      const message = `模型调用失败：${String(error)}`;
      setAssistantMessages((messages) => [
        ...messages,
        { id: ++assistantMessageId, role: "assistant", content: message },
      ]);
      showPopup(message, "error");
    } finally {
      setIsAssistantThinking(false);
    }
  };

  const loadAudioInputDevices = async (requestPermission = false) => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    setIsLoadingDevices(true);
    try {
      if (requestPermission) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((track) => track.stop());
      }
      const devices = await navigator.mediaDevices.enumerateDevices();
      setAudioInputDevices(
        devices
          .filter((device) => device.kind === "audioinput")
          .map((device, index) => ({
            deviceId: device.deviceId,
            label: device.label || `音频输入 ${index + 1}`,
          })),
      );
    } catch (error) {
      console.error("Failed to enumerate audio devices:", error);
      showPopup("无法读取音频输入设备", "error");
    } finally {
      setIsLoadingDevices(false);
    }
  };

  const loadAudioProcesses = async () => {
    setIsLoadingProcesses(true);
    try {
      setAudioProcesses(await invoke<AudioProcessTarget[]>("list_audio_processes"));
    } catch (error) {
      console.error("Failed to list audio processes:", error);
      showPopup("无法读取 Windows 音频进程", "error");
    } finally {
      setIsLoadingProcesses(false);
    }
  };

  const bindProcess = (process: AudioProcessTarget) => {
    setSettings("processAudioPid", process.pid);
    setIsProcessPickerOpen(false);
  };

  const refreshLocalModelStatus = async (modelId: Settings["localSpeechModel"]) => {
    try {
      const downloaded = await invoke<boolean>("check_model_downloaded", { modelId });
      if (settings.localSpeechModel === modelId) {
        setSettings("isModelDownloaded", downloaded);
      }
    } catch (error) {
      console.error("Failed to check local model:", error);
      if (settings.localSpeechModel === modelId) {
        setSettings("isModelDownloaded", false);
      }
    }
  };

  const selectLocalSpeechModel = (modelId: Settings["localSpeechModel"]) => {
    setSettings("localSpeechModel", modelId);
    setSettings("isModelDownloaded", false);
    void refreshLocalModelStatus(modelId);
  };

  const startDownload = async () => {
    setIsDownloading(true);
    setDownloadProgress("0%");
    try {
      await invoke("download_model", { modelId: settings.localSpeechModel });
      setSettings("isModelDownloaded", true);
      showPopup(`${selectedLocalSpeechModel().name} 已准备完成`, "success");
    } catch (error) {
      console.error("Model download failed:", error);
      showPopup(`本地模型下载失败：${String(error)}`, "error");
    } finally {
      setIsDownloading(false);
      setDownloadProgress("");
    }
  };

  const showCaptionWindow = async () => {
    const captionWindow = await Window.getByLabel("caption");
    await captionWindow?.show();
    await captionWindow?.setFocus();
  };

  const toggleRecording = async () => {
    await emit("toggle_recording_shortcut");
  };

  const captureAndAnalyze = async () => {
    if (isCapturingWindow()) return;
    if (isAssistantThinking()) {
      showPopup("请等待当前分析完成", "error");
      return;
    }
    setActiveSection("assistant");
    const capture = await captureVisualContext("foreground");
    if (capture) {
      await sendAssistantMessage("请识别并概括截图中的主要内容。");
    }
  };

  createEffect(() => {
    if (activeSection() === "assistant" && !visualWindowsLoaded && !isLoadingVisualWindows()) {
      visualWindowsLoaded = true;
      void loadVisualWindows();
    }
  });

  createEffect(() => {
    if (activeSection() === "history" && !recordingsLoaded) {
      recordingsLoaded = true;
      void loadRecordings();
    }
  });

  createEffect(() => {
    const transcript = assistantVoice.currentCaption().trim();
    if (
      assistantVoice.isRecording()
      && transcript
      && !transcript.includes("等待音频输入")
      && !transcript.startsWith("正在监听")
    ) {
      setAssistantInput(transcript);
    }
  });

  createEffect(() => {
    const snapshot = JSON.stringify(settings);
    if (!hydrated) return;

    const parsed = JSON.parse(snapshot) as Settings;
    void emit("update-settings", parsed);
    window.clearTimeout(saveTimer);
    setSaveState("saving");
    saveTimer = window.setTimeout(async () => {
      try {
        await invoke("save_settings", { settings: parsed });
        setSaveState("saved");
      } catch (error) {
        console.error("Failed to save settings:", error);
        setSaveState("error");
        showPopup("设置保存失败", "error");
      }
    }, 500);
  });

  onMount(() => {
    void (async () => {
      try {
        unlistenRecording = await listen<{ recording: boolean }>("recording-state", (event) => {
          setIsRecording(event.payload.recording);
        });
        unlistenDownloadProgress = await listen<string>("download-progress", (event) => {
          const percent = event.payload.match(/\d+%/)?.[0];
          if (percent) setDownloadProgress(percent);
        });
        unlistenRecordingUpdated = await listen<RecordingEntry>("recording-updated", (event) => {
          setRecordings((items) => {
            const next = [event.payload, ...items.filter((item) => item.id !== event.payload.id)];
            return next.sort((left, right) => right.startedAt.localeCompare(left.startedAt));
          });
        });
        unlistenCaptureVisual = await listen("capture_visual_shortcut", () => {
          void captureAndAnalyze();
        });
      } catch (error) {
        console.error("Failed to register app event listeners:", error);
      }

      try {
        const loadedSettings = await invoke<LegacySettings>("load_settings");
        const hasProviderConfigs = Boolean(
          loadedSettings.providerConfigs && Object.keys(loadedSettings.providerConfigs).length,
        );
        const providerConfigs = mergeProviderConfigs(loadedSettings.providerConfigs);
        const legacyProvider = loadedSettings.assistantProvider || "dashscope";
        if (!hasProviderConfigs) {
          const legacyConfig = providerConfigs[legacyProvider];
          if (loadedSettings.assistantTextModel?.trim()) {
            legacyConfig.textModel = loadedSettings.assistantTextModel.trim();
          }
          if (loadedSettings.assistantVisionModel?.trim()) {
            legacyConfig.visionModel = loadedSettings.assistantVisionModel.trim();
          }
          if (legacyProvider === "ollama" && loadedSettings.ollamaBaseUrl?.trim()) {
            legacyConfig.baseUrl = loadedSettings.ollamaBaseUrl.trim();
          }
        }
        const migratedSettings: Settings = {
          ...defaultSettings,
          ...loadedSettings,
          speechProvider: loadedSettings.speechProvider || "dashscope",
          assistantTextProvider: loadedSettings.assistantTextProvider || legacyProvider,
          assistantVisionProvider: loadedSettings.assistantVisionProvider || legacyProvider,
          assistantTextModel: loadedSettings.assistantTextModel?.trim()
            || providerConfigs[loadedSettings.assistantTextProvider || legacyProvider].textModel,
          assistantVisionModel: loadedSettings.assistantVisionModel?.trim()
            || providerConfigs[loadedSettings.assistantVisionProvider || legacyProvider].visionModel,
          toggleRecordingHotkey: loadedSettings.toggleRecordingHotkey?.trim()
            || loadedSettings.startRecordingHotkey?.trim()
            || defaultSettings.toggleRecordingHotkey,
          captureVisualHotkey: loadedSettings.captureVisualHotkey?.trim()
            || defaultSettings.captureVisualHotkey,
          providerConfigs,
        };
        setSettings(migratedSettings);
        const localSpeechModel = loadedSettings.localSpeechModel || defaultSettings.localSpeechModel;
        setSettings(
          "isModelDownloaded",
          await invoke<boolean>("check_model_downloaded", { modelId: localSpeechModel }),
        );
      } catch (error) {
        console.error("Failed to load settings:", error);
      }

      hydrated = true;
      await emit("update-settings", unwrap(settings));
      await loadAudioInputDevices(false);
      if (settings.audioSource !== "microphone") await loadAudioProcesses();
    })();
  });

  onCleanup(() => {
    window.clearTimeout(saveTimer);
    unlistenRecording?.();
    unlistenDownloadProgress?.();
    unlistenRecordingUpdated?.();
    unlistenCaptureVisual?.();
    if (assistantVoice.isRecording()) void assistantVoice.stopRecording();
  });

  const generalPanel = () => (
    <div class="panel-content">
      <SectionHeading eyebrow="窗口行为" title="应用与字幕窗口" />
      <div class="setting-list">
        <SettingRow title="字幕窗口始终置顶" description="保持实时字幕位于其他窗口上方">
          <Toggle
            label="字幕窗口始终置顶"
            checked={settings.alwaysOnTop}
            onChange={(checked) => setSettings("alwaysOnTop", checked)}
          />
        </SettingRow>
        <SettingRow title="字幕窗口">
          <button class="secondary-button" onClick={showCaptionWindow}><Eye size={16} />显示字幕</button>
        </SettingRow>
      </div>

      <SectionHeading eyebrow="待机内容" title="默认字幕" />
      <div class="setting-list">
        <SettingRow title="默认文本" description="未开始识别时显示在字幕窗口中" stacked>
          <input
            class="wide-input"
            type="text"
            value={settings.text}
            onInput={(event) => setSettings("text", event.currentTarget.value)}
          />
        </SettingRow>
      </div>

      <SectionHeading eyebrow="运行环境" title="当前状态" />
      <div class="setting-list compact-list">
        <SettingRow title="目标平台"><span class="status-value"><Check size={15} />Windows 11</span></SettingRow>
        <SettingRow title="程序音频"><span class="status-value"><Check size={15} />已支持</span></SettingRow>
        <SettingRow title="界面语言"><span class="muted-value">简体中文</span></SettingRow>
      </div>
    </div>
  );

  const recordingPanel = () => (
    <div class="panel-content">
      <SectionHeading eyebrow="输入" title="录音来源" description="麦克风与 Windows 程序音频使用同一套识别链路" />
      <div class="setting-list">
        <SettingRow title="音频来源">
          <select
            value={settings.audioSource}
            onChange={(event) => {
              const source = event.currentTarget.value as Settings["audioSource"];
              setSettings("audioSource", source);
              if (source !== "microphone") void loadAudioProcesses();
            }}
          >
            <option value="microphone">麦克风输入</option>
            <option value="process">Windows 程序音频</option>
            <option value="microphone_and_process">麦克风 + 程序音频</option>
          </select>
        </SettingRow>

        <Show when={settings.audioSource !== "process"}>
          <SettingRow title="默认设备" description="首次读取设备名称时可能请求麦克风权限">
            <div class="inline-control device-control">
              <select
                value={settings.audioInputDeviceId}
                title={
                  audioInputDevices().find((device) => device.deviceId === settings.audioInputDeviceId)?.label
                  || "系统默认麦克风"
                }
                onChange={(event) => setSettings("audioInputDeviceId", event.currentTarget.value)}
              >
                <option value="">系统默认麦克风</option>
                <For each={audioInputDevices()}>{(device) => <option value={device.deviceId}>{device.label}</option>}</For>
              </select>
              <button class="icon-button" title="刷新设备" onClick={() => loadAudioInputDevices(true)} disabled={isLoadingDevices()}>
                <RefreshCw size={16} classList={{ spinning: isLoadingDevices() }} />
              </button>
            </div>
          </SettingRow>
        </Show>

        <Show when={settings.audioSource !== "microphone"}>
          <SettingRow title="目标程序" description="仅捕获所选进程及其子进程的音频">
            <button
              class="process-select-button"
              onClick={() => {
                setIsProcessPickerOpen(true);
                void loadAudioProcesses();
              }}
            >
              <span>
                <Show when={selectedProcess()} fallback={settings.processAudioPid ? `PID ${settings.processAudioPid}` : "选择程序"}>
                  {selectedProcess()!.name}
                </Show>
              </span>
              <ChevronRight size={16} />
            </button>
          </SettingRow>
          <SettingRow title="捕获方式">
            <span class="status-value"><CircleDot size={15} />Windows Application Loopback</span>
          </SettingRow>
        </Show>
      </div>
    </div>
  );

  const aiPanel = () => (
    <div class="panel-content">
      <SectionHeading eyebrow="语音识别" title="处理方式" description="按网络环境和隐私需求选择在线或本地" />
      <div class="setting-list">
        <SettingRow title="识别方式" stacked>
          <div class="segmented-control">
            <button classList={{ active: settings.modelType === "online" }} onClick={() => setSettings("modelType", "online")}>
              <Cloud size={16} />在线
            </button>
            <button classList={{ active: settings.modelType === "local" }} onClick={() => setSettings("modelType", "local")}>
              <HardDrive size={16} />本地
            </button>
          </div>
        </SettingRow>

        <Show when={settings.modelType === "online"}>
          <SettingRow title="在线服务" description="连接信息由 Provider 统一管理">
            <div class="provider-link-control">
              <span class="status-value" classList={{ warning: !speechProviderConfigured() }}>
                {speechProviderConfigured() ? <Check size={15} /> : <Info size={15} />}
                {speechProviderConfigured() ? "已配置" : "待配置"}
              </span>
              <button class="text-button" onClick={() => openProviderSettings(settings.speechProvider)}>
                <Settings2 size={14} />Provider
              </button>
            </div>
          </SettingRow>
        </Show>

        <Show when={settings.modelType === "local"}>
          <SettingRow title="本地语音模型" description="选择更适合当前语言和性能需求的识别模型">
            <select
              class="local-speech-model-select"
              value={settings.localSpeechModel}
              disabled={isDownloading() || isRecording()}
              title={selectedLocalSpeechModel().name}
              onChange={(event) => selectLocalSpeechModel(event.currentTarget.value as Settings["localSpeechModel"])}
            >
              <For each={localSpeechModels}>{(model) => (
                <option value={model.id}>{model.name}</option>
              )}</For>
            </select>
          </SettingRow>
          <Show when={settings.localSpeechModel === "sense-voice"}>
            <SettingRow title="识别语言" description="指定主要语言可以减少跨语言误判">
              <select
                value={settings.senseVoiceLanguage}
                disabled={isRecording()}
                onChange={(event) => setSettings("senseVoiceLanguage", event.currentTarget.value as SenseVoiceLanguage)}
              >
                <For each={senseVoiceLanguages}>{(language) => (
                  <option value={language.value}>{language.label}</option>
                )}</For>
              </select>
            </SettingRow>
            <div class="recognition-language-help" role="note">
              <Info size={15} />
              <span>多种语言频繁切换时选择“自动检测”；主要识别中文时建议固定为“普通话”，可减少短音频被误判成英文或日语。修改将在下次开始录音时生效。</span>
            </div>
          </Show>
          <SettingRow title="模型文件" description={selectedLocalSpeechModel().description}>
            <Show
              when={settings.isModelDownloaded}
              fallback={
                <button class="primary-button" onClick={startDownload} disabled={isDownloading()}>
                  <Download size={16} />{isDownloading() ? `下载中 ${downloadProgress() || "..."}` : "下载本地资源"}
                </button>
              }
            >
              <span class="status-value"><Check size={15} />{selectedLocalSpeechModel().name} 已就绪</span>
            </Show>
          </SettingRow>
        </Show>
      </div>

      <SectionHeading eyebrow="录音记录" title="文字整理" />
      <div class="setting-list">
        <SettingRow title="录音结束后自动整理" description="补充标点并优化段落，原始文字仍会保留">
          <Toggle
            label="录音结束后自动整理文字"
            checked={settings.autoFormatTranscripts}
            onChange={(checked) => setSettings("autoFormatTranscripts", checked)}
          />
        </SettingRow>
        <Show when={settings.autoFormatTranscripts}>
          <SettingRow title="使用模型" description="跟随 AI 助手中的文字模型">
            <span class="muted-value">{assistantTextProvider().name} · {settings.assistantTextModel || "未选择"}</span>
          </SettingRow>
        </Show>
      </div>

      <SectionHeading eyebrow="本地分析" title="说话人分离" />
      <div class="setting-list">
        <SettingRow title="录音结束后自动分离" description="CPU 本地处理 · 匿名说话人 · 模型约 46 MB">
          <Toggle
            label="录音结束后自动进行说话人分离"
            checked={settings.speakerDiarizationEnabled}
            onChange={(checked) => setSettings("speakerDiarizationEnabled", checked)}
          />
        </SettingRow>
      </div>

      <SectionHeading eyebrow="输出" title="处理流程" />
      <div class="pipeline">
        <span>音频输入</span><ChevronRight size={15} />
        <span>{settings.modelType === "online" ? "在线处理" : "本地处理"}</span>
        <ChevronRight size={15} /><span>实时字幕</span>
      </div>
    </div>
  );

  const providerPanel = () => (
    <div class="panel-content provider-panel">
      <SectionHeading
        eyebrow="Provider"
        title="服务商设置"
        description="统一维护连接信息，并读取这个服务可用的模型"
      />
      <div class="setting-list">
        <SettingRow title="服务商" description="选择要查看和编辑的配置">
          <select
            class="provider-profile-select"
            value={selectedProviderId()}
            onChange={(event) => {
              setSelectedProviderId(event.currentTarget.value as ProviderId);
              setShowProviderKey(false);
              setManualProviderModel("");
            }}
          >
            <For each={providerDefinitions}>{(provider) => (
              <option value={provider.id}>{provider.name}</option>
            )}</For>
          </select>
        </SettingRow>
        <SettingRow title="运行位置">
          <span class="status-value">
            {selectedProviderDefinition().mode === "online" ? <Cloud size={15} /> : <HardDrive size={15} />}
            {selectedProviderDefinition().mode === "online" ? "在线" : "本地"}
          </span>
        </SettingRow>
        <SettingRow title="可用范围">
          <span class="provider-capabilities">
            文字{selectedProviderDefinition().supportsVision ? " · 视觉" : ""}{selectedProviderDefinition().supportsSpeech ? " · 语音识别" : ""}
          </span>
        </SettingRow>
        <SettingRow title="服务地址" description="该 Provider 的统一访问地址">
          <input
            class="provider-field"
            type="text"
            value={selectedProviderConfig().baseUrl}
            placeholder="输入服务地址"
            onInput={(event) => updateSelectedProvider("baseUrl", event.currentTarget.value)}
          />
        </SettingRow>
        <Show when={selectedProviderDefinition().mode === "online"}>
          <SettingRow title="访问密钥" description={selectedProviderId() === "dashscope" && dashscopeEnvApiKey ? "留空时继续使用环境配置" : "仅在本机设置中保存"}>
            <div class="provider-secret-control">
              <input
                class="provider-field"
                type={showProviderKey() ? "text" : "password"}
                value={selectedProviderConfig().apiKey}
                placeholder={selectedProviderId() === "dashscope" && dashscopeEnvApiKey ? "已从环境读取" : "输入访问密钥"}
                autocomplete="off"
                onInput={(event) => updateSelectedProvider("apiKey", event.currentTarget.value)}
              />
              <button
                class="icon-button"
                title={showProviderKey() ? "隐藏访问密钥" : "显示访问密钥"}
                onClick={() => setShowProviderKey((visible) => !visible)}
              >
                {showProviderKey() ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </SettingRow>
        </Show>
        <SettingRow
          title="可用模型"
          description={selectedProviderConfig().availableModels.length
            ? `已保存 ${selectedProviderConfig().availableModels.length} 个模型`
            : "尚未获取模型列表"}
        >
          <button class="secondary-button" onClick={refreshProviderModels} disabled={isLoadingProviderModels()}>
            <RefreshCw size={15} classList={{ spinning: isLoadingProviderModels() }} />
            {isLoadingProviderModels() ? "获取中" : "获取模型"}
          </button>
        </SettingRow>
        <SettingRow title="手动添加" description="用于不提供模型列表的兼容服务">
          <div class="provider-model-add">
            <input
              class="provider-field"
              type="text"
              value={manualProviderModel()}
              placeholder="模型名称"
              onInput={(event) => setManualProviderModel(event.currentTarget.value)}
              onKeyDown={(event) => event.key === "Enter" && addManualProviderModel()}
            />
            <button class="icon-button" title="添加模型" onClick={addManualProviderModel} disabled={!manualProviderModel().trim()}>
              <Plus size={16} />
            </button>
          </div>
        </SettingRow>
        <SettingRow title="恢复默认" description="重置当前 Provider 的地址和模型">
          <button class="text-button" onClick={resetSelectedProvider}><RotateCcw size={14} />重置</button>
        </SettingRow>
      </div>
    </div>
  );

  const assistantPanel = () => (
    <div class="panel-content assistant-panel">
      <SectionHeading eyebrow="模型" title="对话模型" description="文字和视觉任务可以使用不同的 Provider" />
      <div class="setting-list">
        <SettingRow title="文字模型" description="先选择 Provider，再选择模型" stacked>
          <div class="model-binding-control">
            <select
              value={settings.assistantTextProvider}
              aria-label="文字模型 Provider"
              onChange={(event) => selectAssistantProvider("text", event.currentTarget.value as ProviderId)}
            >
              <For each={providerDefinitions}>{(provider) => (
                <option value={provider.id}>{provider.name}</option>
              )}</For>
            </select>
            <Show
              when={providerModels(settings.assistantTextProvider, settings.assistantTextModel).length > 0}
              fallback={
                <input
                  type="text"
                  value={settings.assistantTextModel}
                  placeholder="输入模型名称"
                  onInput={(event) => setSettings("assistantTextModel", event.currentTarget.value)}
                />
              }
            >
              <select
                value={settings.assistantTextModel}
                aria-label="文字模型"
                title={settings.assistantTextModel}
                onChange={(event) => setSettings("assistantTextModel", event.currentTarget.value)}
              >
                <For each={providerModels(settings.assistantTextProvider, settings.assistantTextModel)}>{(model) => (
                  <option value={model}>{model}</option>
                )}</For>
              </select>
            </Show>
          </div>
        </SettingRow>
        <SettingRow title="视觉模型" description="可以与文字模型使用不同 Provider" stacked>
          <div class="model-binding-control">
            <select
              value={settings.assistantVisionProvider}
              aria-label="视觉模型 Provider"
              onChange={(event) => selectAssistantProvider("vision", event.currentTarget.value as ProviderId)}
            >
              <For each={providerDefinitions.filter((provider) => provider.supportsVision)}>{(provider) => (
                <option value={provider.id}>{provider.name}</option>
              )}</For>
            </select>
            <Show
              when={providerModels(settings.assistantVisionProvider, settings.assistantVisionModel).length > 0}
              fallback={
                <input
                  type="text"
                  value={settings.assistantVisionModel}
                  placeholder="输入模型名称"
                  onInput={(event) => setSettings("assistantVisionModel", event.currentTarget.value)}
                />
              }
            >
              <select
                value={settings.assistantVisionModel}
                aria-label="视觉模型"
                title={settings.assistantVisionModel}
                onChange={(event) => setSettings("assistantVisionModel", event.currentTarget.value)}
              >
                <For each={providerModels(settings.assistantVisionProvider, settings.assistantVisionModel)}>{(model) => (
                  <option value={model}>{model}</option>
                )}</For>
              </select>
            </Show>
          </div>
        </SettingRow>
        <SettingRow title="Provider 设置" description="地址、密钥和可用模型在统一入口维护">
          <button class="text-button" onClick={() => openProviderSettings(settings.assistantTextProvider)}>
            <Settings2 size={14} />打开设置
          </button>
        </SettingRow>
      </div>

      <SectionHeading
        eyebrow="视觉上下文"
        title="理解当前页面或指定窗口"
        description="截图只会在你发送问题且启用视觉上下文时交给所选模型"
      />
      <div class="visual-capture-toolbar">
        <button
          class="primary-button"
          disabled={isCapturingWindow()}
          onClick={() => captureVisualContext("foreground")}
        >
          <Eye size={16} />{isCapturingWindow() ? "正在捕获" : "捕获当前页面"}
        </button>
        <div class="visual-window-control">
          <select
            value={visualWindowId()}
            title={selectedVisualWindow()?.title || "选择窗口"}
            onChange={(event) => setVisualWindowId(event.currentTarget.value)}
          >
            <option value="">选择指定窗口</option>
            <For each={visualWindows()}>{(window) => (
              <option value={window.id}>
                {window.processName} · {window.title}{window.minimized ? "（已最小化）" : ""}
              </option>
            )}</For>
          </select>
          <button
            class="secondary-button"
            disabled={!visualWindowId() || isCapturingWindow()}
            onClick={() => captureVisualContext("selected")}
          >
            <BoxSelect size={16} />捕获
          </button>
          <button class="icon-button" title="刷新窗口列表" onClick={loadVisualWindows} disabled={isLoadingVisualWindows()}>
            <RefreshCw size={16} classList={{ spinning: isLoadingVisualWindows() }} />
          </button>
        </div>
      </div>

      <Show
        when={capturedWindow()}
        fallback={<div class="visual-empty"><MonitorUp size={24} /><span>尚未捕获窗口，仍可直接进行文字或语音对话</span></div>}
      >
        {(capture) => (
          <div class="visual-preview-card">
            <div class="visual-preview-meta">
              <div>
                <strong>{capture().windowTitle || capture().processName}</strong>
                <span>{capture().processName} · {capture().width} × {capture().height}</span>
              </div>
              <div class="visual-preview-actions">
                <span>发送时携带</span>
                <Toggle label="发送时携带窗口截图" checked={useVisualContext()} onChange={setUseVisualContext} />
                <button class="icon-button" title="移除截图" onClick={() => setCapturedWindow(null)}><X size={16} /></button>
              </div>
            </div>
            <img src={capture().imageDataUrl} alt={`窗口截图：${capture().windowTitle}`} />
          </div>
        )}
      </Show>

      <SectionHeading
        eyebrow="交互"
        title="连续对话"
        description="语音听写沿用上方语音识别设置；Ctrl + Enter 可快速发送"
      />
      <div class="assistant-chat-toolbar">
        <div class="quick-prompts">
          <button disabled={!capturedWindow() || isAssistantThinking()} onClick={() => sendAssistantMessage("概括当前页面的主要内容和重点。")}>概括页面</button>
          <button disabled={!capturedWindow() || isAssistantThinking()} onClick={() => sendAssistantMessage("提取并整理当前页面中可见的文字。")}>提取文字</button>
          <button disabled={!capturedWindow() || isAssistantThinking()} onClick={() => sendAssistantMessage("分析当前界面，告诉我下一步最合适的操作。")}>下一步建议</button>
        </div>
        <button
          class="text-button"
          onClick={() => setAssistantMessages([{ id: ++assistantMessageId, role: "assistant", content: "新对话已开始。" }])}
        >
          <RotateCcw size={14} />清空
        </button>
      </div>
      <div class="assistant-chat">
        <For each={assistantMessages()}>{(message) => (
          <div class="assistant-message" classList={{ user: message.role === "user", assistant: message.role === "assistant" }}>
            <span>{message.role === "user" ? "你" : "AI"}</span>
            <p>{message.content}</p>
          </div>
        )}</For>
        <Show when={isAssistantThinking()}>
          <div class="assistant-message assistant thinking"><span>AI</span><p>正在思考…</p></div>
        </Show>
      </div>
      <div class="assistant-composer">
        <textarea
          rows="3"
          placeholder={assistantVoice.isRecording() ? "正在听写，请说出你的问题…" : "输入问题，或点击语音输入开始说话"}
          value={assistantInput()}
          onInput={(event) => setAssistantInput(event.currentTarget.value)}
          onKeyDown={(event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
              event.preventDefault();
              void sendAssistantMessage();
            }
          }}
        />
        <div class="assistant-composer-actions">
          <button
            classList={{ "voice-input-button": true, active: assistantVoice.isRecording() }}
            disabled={assistantVoice.isTransitioning() || isAssistantThinking()}
            onClick={toggleAssistantVoice}
          >
            {assistantVoice.isRecording() ? <Square size={15} fill="currentColor" /> : <Mic2 size={16} />}
            {assistantVoice.isRecording() ? "停止听写" : "语音输入"}
          </button>
          <span classList={{ "assistant-input-status": true, error: Boolean(assistantVoice.error()) }}>
            {assistantVoice.error() || (capturedWindow() && useVisualContext()
              ? `${assistantVisionProvider().name} · ${settings.assistantVisionModel || "未选择"}`
              : `${assistantTextProvider().name} · ${settings.assistantTextModel || "未选择"}`)}
          </span>
          <button class="primary-button" disabled={!assistantInput().trim() || isAssistantThinking()} onClick={() => sendAssistantMessage()}>
            <ChevronRight size={16} />发送
          </button>
        </div>
      </div>
    </div>
  );

  const historyPanel = () => (
    <div class="panel-content recording-history-panel">
      <Show
        when={selectedRecording()}
        fallback={
          <>
            <SectionHeading eyebrow="记录" title="录音记录" description="每次录音结束后会保留音频与完整文字" />
            <div class="recording-list-toolbar">
              <span>{recordings().length} 条记录</span>
              <button class="icon-button" title="刷新录音记录" onClick={loadRecordings} disabled={isLoadingRecordings()}>
                <RefreshCw size={16} classList={{ spinning: isLoadingRecordings() }} />
              </button>
            </div>
            <div class="recording-list">
              <For
                each={recordings()}
                fallback={
                  <div class="recording-empty">
                    <ListMusic size={28} />
                    <strong>{isLoadingRecordings() ? "正在读取" : "还没有录音记录"}</strong>
                  </div>
                }
              >
                {(recording) => (
                  <button class="recording-row" onClick={() => setSelectedRecordingId(recording.id)}>
                    <span class="recording-row-icon"><FileText size={18} /></span>
                    <span class="recording-row-main">
                      <strong>{recording.title}</strong>
                      <span>
                        {formatRecordingTime(recording.startedAt)} · {formatRecordingDuration(recording.durationMs)} · {formatAudioSource(recording.audioSource)}
                      </span>
                    </span>
                    <span class="recording-row-status">
                      {recording.diarizationStatus === "pending" ? "音色分析中" : recording.formattingStatus === "pending" ? "整理中" : recording.reportStatus === "ready" ? "有报告" : "已保存"}
                      <ChevronRight size={16} />
                    </span>
                  </button>
                )}
              </For>
            </div>
          </>
        }
      >
        {(recording) => (
          <div class="recording-detail">
            <div class="recording-detail-nav">
              <button class="icon-button" title="返回录音列表" onClick={() => setSelectedRecordingId("")}><ArrowLeft size={18} /></button>
              <span>{formatRecordingTime(recording().startedAt)}</span>
            </div>
            <SectionHeading
              eyebrow={formatAudioSource(recording().audioSource)}
              title={recording().title}
              description={`${formatRecordingDuration(recording().durationMs)} · ${recording().textModel || "未使用文字整理"}`}
            />

            <audio ref={recordingAudioElement} class="recording-audio" controls preload="metadata" src={convertFileSrc(recording().audioPath)} />

            <section class="recording-section">
              <div class="recording-section-heading">
                <div><span class="section-eyebrow">本地分析</span><h3>说话人分离</h3></div>
                <Show
                  when={recording().diarizationStatus === "pending"}
                  fallback={
                    <button class="secondary-button" disabled={isDiarizing()} onClick={diarizeSelectedRecording}>
                      <RefreshCw size={15} classList={{ spinning: isDiarizing() }} />
                      {recording().diarizationStatus === "ready" ? "重新分析" : "开始分析"}
                    </button>
                  }
                >
                  <span class="processing-label"><RefreshCw size={14} class="spinning" />正在本地分析</span>
                </Show>
              </div>
              <Show when={recording().diarizationError}>
                <div class="inline-error">说话人分离失败：{recording().diarizationError}</div>
              </Show>
              <Show
                when={recording().speakerSegments.length > 0}
                fallback={
                  <Show when={recording().diarizationStatus === "ready"}>
                    <div class="report-empty">没有检测到清晰的发言片段</div>
                  </Show>
                }
              >
                <div class="speaker-summary">检测到 {recording().speakerCount} 位说话人</div>
                <div class="speaker-timeline">
                  <For each={recording().speakerSegments}>{(segment) => (
                    <button class="speaker-segment" onClick={() => playSpeakerSegment(segment)}>
                      <span class={`speaker-marker speaker-${segment.speaker % 6}`} />
                      <strong>说话人 {segment.speaker + 1}</strong>
                      <span>{formatRecordingDuration(segment.startMs)} - {formatRecordingDuration(segment.endMs)}</span>
                      <Play size={14} fill="currentColor" />
                    </button>
                  )}</For>
                </div>
              </Show>
            </section>

            <section class="recording-section">
              <div class="recording-section-heading">
                <div><span class="section-eyebrow">文字</span><h3>{recording().formattedTranscript ? "整理后的文字" : "原始文字"}</h3></div>
                <Show when={recording().formattingStatus === "pending"}><span class="processing-label"><RefreshCw size={14} class="spinning" />整理中</span></Show>
              </div>
              <Show when={recording().formattingError}>
                <div class="inline-error">文字整理失败：{recording().formattingError}</div>
              </Show>
              <div class="recording-copy">
                {recording().formattedTranscript || recording().rawTranscript || "本次录音没有识别到文字"}
              </div>
              <Show when={recording().formattedTranscript}>
                <details class="raw-transcript"><summary>查看原始文字</summary><div>{recording().rawTranscript}</div></details>
              </Show>
            </section>

            <section class="recording-section">
              <div class="recording-section-heading">
                <div><span class="section-eyebrow">报告</span><h3>汇总报告</h3></div>
                <button
                  class="primary-button"
                  disabled={isGeneratingReport() || recording().reportStatus === "generating" || !recording().rawTranscript.trim()}
                  onClick={generateRecordingReport}
                >
                  <Sparkles size={16} />
                  {isGeneratingReport() || recording().reportStatus === "generating" ? "生成中" : recording().report ? "重新生成" : "生成报告"}
                </button>
              </div>
              <Show when={recording().reportError}>
                <div class="inline-error">报告生成失败：{recording().reportError}</div>
              </Show>
              <Show when={recording().report} fallback={<div class="report-empty">尚未生成汇总报告</div>}>
                <pre class="recording-report">{recording().report}</pre>
              </Show>
            </section>

            <div class="recording-share-actions">
              <span><Share2 size={16} />分享与导出</span>
              <button class="secondary-button" disabled={Boolean(exportingFormat())} onClick={() => exportRecording("markdown")}>
                <FileDown size={16} />{exportingFormat() === "markdown" ? "生成中" : "Markdown"}
              </button>
              <button class="secondary-button" disabled={Boolean(exportingFormat())} onClick={() => exportRecording("pdf")}>
                <FileDown size={16} />{exportingFormat() === "pdf" ? "生成中" : "PDF"}
              </button>
            </div>
          </div>
        )}
      </Show>
    </div>
  );

  const memoryPanel = () => (
    <div class="panel-content">
      <SectionHeading eyebrow="数据" title="本地存储" description="设置、模型与录音记录均保存在当前设备" />
      <div class="setting-list">
        <SettingRow title="设置存储" description="仅保存在当前设备">
          <span class="status-value"><Check size={15} />正常</span>
        </SettingRow>
        <SettingRow title="本地模型" description={selectedLocalSpeechModel().name}>
          <span class="status-value" classList={{ warning: !settings.isModelDownloaded }}>
            {settings.isModelDownloaded ? <Check size={15} /> : <Info size={15} />}
            {settings.isModelDownloaded ? "已缓存" : "未下载"}
          </span>
        </SettingRow>
        <SettingRow title="录音记录" description="包含音频、文字和已生成的报告">
          <span class="status-value"><Check size={15} />已启用</span>
        </SettingRow>
      </div>
    </div>
  );

  const captionPanel = () => {
    const colors = ["#FFFFFF", "#FFD43B", "#20D997", "#22C7E5"];
    const backgrounds = ["#111318", "#1F2937", "#0F172A"];
    return (
      <div class="panel-content">
        <SectionHeading eyebrow="实时预览" title="字幕外观" />
        <div class="setting-list caption-controls">
          <SettingRow title="显示位置" description="顶部和底部会在当前显示器中自动居中" stacked>
            <div class="mode-picker position-picker">
              <button classList={{ active: settings.captionPosition === "top" }} onClick={() => setSettings("captionPosition", "top")}>
                <ArrowUpToLine size={17} /><span>顶部</span>
              </button>
              <button classList={{ active: settings.captionPosition === "bottom" }} onClick={() => setSettings("captionPosition", "bottom")}>
                <ArrowDownToLine size={17} /><span>底部</span>
              </button>
              <button classList={{ active: settings.captionPosition === "free" }} onClick={() => setSettings("captionPosition", "free")}>
                <Move size={17} /><span>自由拖动</span>
              </button>
            </div>
          </SettingRow>
          <SettingRow title="字体大小">
            <div class="range-control"><span>14</span><input type="range" min="14" max="36" value={settings.fontSize} onInput={(event) => setSettings("fontSize", Number(event.currentTarget.value))} /><output>{settings.fontSize}px</output></div>
          </SettingRow>
          <div
            class="caption-preview"
            classList={{
              glass: settings.captionBackgroundStyle === "glass",
              solid: settings.captionBackgroundStyle === "solid",
              outline: settings.captionBackgroundStyle === "outline",
            }}
            style={{
              color: settings.fontColor,
              background: previewBackground(),
              "font-size": `${Math.min(settings.fontSize, 34)}px`,
            }}
          >
            {settings.text || "字幕会显示在这里"}
          </div>
          <SettingRow title="透明度">
            <div class="range-control"><span>20%</span><input type="range" min="0.2" max="1" step="0.05" value={settings.opacity} onInput={(event) => setSettings("opacity", Number(event.currentTarget.value))} /><output>{Math.round(settings.opacity * 100)}%</output></div>
          </SettingRow>
          <SettingRow title="字体颜色">
            <div class="swatch-group">
              <For each={colors}>{(color) => <button aria-label={`字体颜色 ${color}`} class="color-swatch" classList={{ active: settings.fontColor.toUpperCase() === color }} style={{ background: color }} onClick={() => setSettings("fontColor", color)} />}</For>
              <label class="custom-color" title="自定义字体颜色"><input type="color" value={settings.fontColor} onInput={(event) => setSettings("fontColor", event.currentTarget.value)} /><Sparkles size={15} /></label>
            </div>
          </SettingRow>
          <SettingRow title="背景颜色">
            <div class="swatch-group">
              <For each={backgrounds}>{(color) => <button aria-label={`背景颜色 ${color}`} class="color-swatch" classList={{ active: settings.backgroundColor.toUpperCase() === color }} style={{ background: color }} onClick={() => setSettings("backgroundColor", color)} />}</For>
              <label class="custom-color" title="自定义背景颜色"><input type="color" value={settings.backgroundColor} onInput={(event) => setSettings("backgroundColor", event.currentTarget.value)} /><Sparkles size={15} /></label>
            </div>
          </SettingRow>
          <SettingRow title="背景样式" stacked>
            <div class="mode-picker background-style-picker">
              <button classList={{ active: settings.captionBackgroundStyle === "glass" }} onClick={() => setSettings("captionBackgroundStyle", "glass")}>
                <Layers3 size={17} /><span>深色毛玻璃</span>
              </button>
              <button classList={{ active: settings.captionBackgroundStyle === "solid" }} onClick={() => setSettings("captionBackgroundStyle", "solid")}>
                <Square size={17} /><span>纯色背景</span>
              </button>
              <button classList={{ active: settings.captionBackgroundStyle === "outline" }} onClick={() => setSettings("captionBackgroundStyle", "outline")}>
                <BoxSelect size={17} /><span>无背景仅描边</span>
              </button>
            </div>
          </SettingRow>
          <SettingRow title="始终置顶">
            <Toggle label="字幕窗口始终置顶" checked={settings.alwaysOnTop} onChange={(checked) => setSettings("alwaysOnTop", checked)} />
          </SettingRow>
        </div>
      </div>
    );
  };

  const shortcutsPanel = () => (
    <div class="panel-content">
      <SectionHeading eyebrow="全局快捷键" title="快速操作" description="快捷键在应用位于后台时同样生效" />
      <div class="shortcut-table">
        <div class="shortcut-head"><span>功能</span><span>快捷键</span><span>操作</span></div>
        <div class="shortcut-row">
          <span class="shortcut-name"><Mic2 size={16} />开始 / 结束录音</span>
          <input type="text" value={settings.toggleRecordingHotkey} onInput={(event) => setSettings("toggleRecordingHotkey", event.currentTarget.value)} />
          <button class="text-button" onClick={() => setSettings("toggleRecordingHotkey", defaultSettings.toggleRecordingHotkey)}><RotateCcw size={14} />恢复默认</button>
        </div>
        <div class="shortcut-row">
          <span class="shortcut-name"><BoxSelect size={16} />截图并分析</span>
          <input type="text" value={settings.captureVisualHotkey} onInput={(event) => setSettings("captureVisualHotkey", event.currentTarget.value)} />
          <button class="text-button" onClick={() => setSettings("captureVisualHotkey", defaultSettings.captureVisualHotkey)}><RotateCcw size={14} />恢复默认</button>
        </div>
      </div>
    </div>
  );

  const aboutPanel = () => (
    <div class="about-panel">
      <div class="app-mark"><Mic2 size={38} strokeWidth={1.8} /></div>
      <h2>VoiceAI</h2>
      <span class="about-version">v{VERSION}</span>
      <p>实时语音识别与 Windows 程序音频字幕</p>
      <small>本地优先 · 无需虚拟声卡</small>
    </div>
  );

  const renderPanel = () => {
    switch (activeSection()) {
      case "general": return generalPanel();
      case "recording": return recordingPanel();
      case "history": return historyPanel();
      case "ai": return aiPanel();
      case "providers": return providerPanel();
      case "assistant": return assistantPanel();
      case "memory": return memoryPanel();
      case "captions": return captionPanel();
      case "shortcuts": return shortcutsPanel();
      case "about": return aboutPanel();
    }
  };

  return (
    <>
      <main class="app-shell">
        <aside class="sidebar">
          <div class="brand-block">
            <div class="brand-icon"><Mic2 size={19} /></div>
            <div><strong>VoiceAI</strong><span>设置</span></div>
          </div>
          <nav class="sidebar-nav" aria-label="设置导航">
            <For each={navItems}>{(item) => (
              <button classList={{ active: activeSection() === item.id }} onClick={() => setActiveSection(item.id)} title={item.description}>
                <Dynamic component={item.icon} size={18} strokeWidth={1.8} />
                <span>{item.label}</span>
              </button>
            )}</For>
          </nav>
        </aside>

        <section class="workspace">
          <header class="workspace-header">
            <div><span class="workspace-kicker">设置</span><h1>{activeItem().label}</h1><p>{activeItem().description}</p></div>
            <div class="header-actions">
              <button class="icon-button" title="显示字幕窗口" onClick={showCaptionWindow}><MonitorUp size={18} /></button>
              <button classList={{ "record-button": true, active: isRecording() }} onClick={toggleRecording}>
                {isRecording() ? <Square size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
                {isRecording() ? "停止录音" : "开始录音"}
              </button>
            </div>
          </header>

          <div class="workspace-scroll">{renderPanel()}</div>

          <footer class="status-bar">
            <span classList={{ "status-dot": true, recording: isRecording() }} />
            <span>{isRecording() ? "录音中" : "待机中"}</span>
            <i />
            <span>{saveState() === "saving" ? "正在保存" : saveState() === "error" ? "保存失败" : "设置已保存"}</span>
            <i />
            <span>v{VERSION}</span>
          </footer>
        </section>
      </main>

      <Show when={isProcessPickerOpen()}>
        <Portal mount={document.getElementById("portal")!}>
          <div class="modal-backdrop" onMouseDown={() => setIsProcessPickerOpen(false)}>
            <section class="process-picker" onMouseDown={(event) => event.stopPropagation()}>
              <header class="process-picker-header">
                <div><span class="section-eyebrow">Windows 音频会话</span><h2>选择目标程序</h2></div>
                <button class="icon-button" title="关闭" onClick={() => setIsProcessPickerOpen(false)}><X size={18} /></button>
              </header>
              <div class="process-picker-toolbar">
                <label class="search-field"><Search size={16} /><input type="search" placeholder="搜索程序、窗口标题或 PID" value={processSearch()} onInput={(event) => setProcessSearch(event.currentTarget.value)} /></label>
                <button class="secondary-button" onClick={loadAudioProcesses} disabled={isLoadingProcesses()}><RefreshCw size={16} classList={{ spinning: isLoadingProcesses() }} />刷新</button>
              </div>
              <div class="process-list">
                <For each={filteredProcesses()} fallback={<div class="empty-state">没有找到可用程序</div>}>
                  {(process) => (
                    <button class="process-row" classList={{ selected: settings.processAudioPid === process.pid }} onClick={() => bindProcess(process)}>
                      <span class="process-avatar">{process.name.slice(0, 1).toUpperCase()}</span>
                      <span class="process-main">
                        <span class="process-heading"><strong>{process.name}</strong><Show when={process.hasAudioSession}><em>正在发声</em></Show></span>
                        <span>{process.windowTitle || process.executablePath || "后台进程"}</span>
                      </span>
                      <span class="process-meta">PID {process.pid}<Show when={settings.processAudioPid === process.pid}><Check size={16} /></Show></span>
                    </button>
                  )}
                </For>
              </div>
            </section>
          </div>
        </Portal>
      </Show>

      <Show when={popupInfo.show}>
        <Portal mount={document.getElementById("portal")!}>
          <div class={`toast ${popupInfo.type}`}>{popupInfo.type === "success" ? <Check size={16} /> : <Info size={16} />}{popupInfo.message}</div>
        </Portal>
      </Show>
    </>
  );
}

export default App;
