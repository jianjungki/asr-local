import { createSignal } from 'solid-js';
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { DashscopeWebSocketClient, type DashscopeConfig } from '../lib/dashscopeClient';
import type { SenseVoiceLanguage } from '../lib/languages';

type RecordingOptions = {
    modelType: "online" | "local";
    localSpeechModel?: "paraformer-streaming" | "sense-voice" | "zipformer-zh";
    senseVoiceLanguage?: SenseVoiceLanguage;
    audioSource?: "microphone" | "process" | "microphone_and_process";
    audioInputDeviceId?: string;
    processAudioPid?: number;
    onlineApiKey?: string;
    saveAudio?: boolean;
};

export type CaptionResult = {
    text: string;
    isFinal: boolean;
    sequence: number;
    receivedAt: number;
};

type LocalTranscriptResult = {
    text: string;
    isFinal: boolean;
};

export function useSpeechRecognition() {
    const [isRecording, setIsRecording] = createSignal(false);
    const [currentCaption, setCurrentCaption] = createSignal('');
    const [captionResult, setCaptionResult] = createSignal<CaptionResult | null>(null);
    const [error, setError] = createSignal<string | null>(null);
    const [isTransitioning, setIsTransitioning] = createSignal(false);
    const [processAudioDebug, setProcessAudioDebug] = createSignal('');

    let audioContext: AudioContext | null = null;
    let scriptProcessor: ScriptProcessorNode | null = null;
    let mediaStream: MediaStream | null = null;
    let dashscopeClient: DashscopeWebSocketClient | null = null;
    let processTranscriptUnlisten: UnlistenFn | null = null;
    let processChunkUnlisten: UnlistenFn | null = null;
    let processStatsUnlisten: UnlistenFn | null = null;
    let processErrorUnlisten: UnlistenFn | null = null;
    let activeAudioSource: "microphone" | "process" | "microphone_and_process" = "microphone";
    let activeModelType: "online" | "local" = "online";
    let processChunksSent = 0;
    let captionSequence = 0;
    let localAudioQueue: Promise<void> = Promise.resolve();
    let recordingAudioQueue: Promise<void> = Promise.resolve();
    let recordingAudioBuffer: number[] = [];
    let processAudioBuffer: number[] = [];
    let processAudioReadOffset = 0;
    let shouldSaveAudio = false;

    const flushRecordingAudio = () => {
        if (!shouldSaveAudio || activeAudioSource === "process" || recordingAudioBuffer.length === 0) {
            return recordingAudioQueue;
        }
        const samples = recordingAudioBuffer;
        recordingAudioBuffer = [];
        recordingAudioQueue = recordingAudioQueue
            .then(() => invoke<void>('append_recording_audio', { samples }))
            .catch((writeError) => {
                console.error("Failed to persist microphone audio:", writeError);
            });
        return recordingAudioQueue;
    };

    const queueRecordingAudio = (samples: Float32Array) => {
        if (!shouldSaveAudio || activeAudioSource === "process") return;
        recordingAudioBuffer.push(...samples);
        if (recordingAudioBuffer.length >= 16_000) void flushRecordingAudio();
    };

    const queueProcessAudio = (samples: number[]) => {
        processAudioBuffer.push(...samples);

        const maxBufferedSamples = 32_000;
        const unreadSamples = processAudioBuffer.length - processAudioReadOffset;
        if (unreadSamples > maxBufferedSamples) {
            processAudioReadOffset += unreadSamples - maxBufferedSamples;
        }
        if (processAudioReadOffset >= 16_000) {
            processAudioBuffer = processAudioBuffer.slice(processAudioReadOffset);
            processAudioReadOffset = 0;
        }
    };

    const mixWithProcessAudio = (microphoneSamples: Float32Array) => {
        const mixed = new Float32Array(microphoneSamples.length);
        for (let index = 0; index < microphoneSamples.length; index += 1) {
            const processSample = processAudioReadOffset < processAudioBuffer.length
                ? processAudioBuffer[processAudioReadOffset++]
                : 0;
            mixed[index] = Math.max(-1, Math.min(1, microphoneSamples[index] + processSample));
        }

        if (processAudioReadOffset >= 16_000) {
            processAudioBuffer = processAudioBuffer.slice(processAudioReadOffset);
            processAudioReadOffset = 0;
        }
        return mixed;
    };

    const publishCaption = (text: string, isFinal: boolean) => {
        const normalized = text
            .replace(/<unk>/gi, "")
            .replace(/\s{2,}/g, " ")
            .trim();
        if (!normalized) return;

        const receivedAt = Date.now();
        setCurrentCaption(normalized);
        setCaptionResult({
            text: normalized,
            isFinal,
            sequence: ++captionSequence,
            receivedAt,
        });
    };

    const startRecording = async (options: RecordingOptions | "online" | "local") => {
        if (isTransitioning() || isRecording()) {
            console.log("Already recording or in transition, ignoring start request.");
            return false;
        }

        setIsTransitioning(true);
        try {
            const modelType = typeof options === "string" ? options : options.modelType;
            const localSpeechModel = typeof options === "string" ? "paraformer-streaming" : options.localSpeechModel || "paraformer-streaming";
            const senseVoiceLanguage = typeof options === "string" ? "auto" : options.senseVoiceLanguage || "auto";
            const audioSource = typeof options === "string" ? "microphone" : options.audioSource || "microphone";
            const audioInputDeviceId = typeof options === "string" ? "" : options.audioInputDeviceId || "";
            const processAudioPid = typeof options === "string" ? 0 : options.processAudioPid || 0;
            const onlineApiKey = typeof options === "string" ? "" : options.onlineApiKey || "";
            shouldSaveAudio = typeof options === "string" ? false : options.saveAudio === true;

            setError(null);
            setProcessAudioDebug('');
            captionSequence = 0;
            setCaptionResult(null);
            processChunksSent = 0;
            activeAudioSource = audioSource;
            activeModelType = modelType;
            localAudioQueue = Promise.resolve();
            recordingAudioQueue = Promise.resolve();
            recordingAudioBuffer = [];
            processAudioBuffer = [];
            processAudioReadOffset = 0;
            console.log(`Starting recording with model type: ${modelType}, audio source: ${audioSource}`);

            if (modelType === "local") {
                setCurrentCaption('正在准备本地识别...');
                await invoke('prepare_local_model', { modelId: localSpeechModel, senseVoiceLanguage });
            }

            if (audioSource === "microphone_and_process") {
                if (!processAudioPid || processAudioPid <= 0) {
                    throw new Error("请先选择要捕获音频的 Windows 程序");
                }

                const config: DashscopeConfig = {
                    apiKey: onlineApiKey || import.meta.env.VITE_DASHSCOPE_API_KEY || '',
                };
                if (modelType === "online" && !config.apiKey) {
                    throw new Error('在线服务访问密钥未配置');
                }

                const audioConstraints: boolean | MediaTrackConstraints = audioInputDeviceId
                    ? { deviceId: { exact: audioInputDeviceId } }
                    : true;
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(mediaStream);
                scriptProcessor = audioContext.createScriptProcessor(modelType === "local" ? 1024 : 4096, 1, 1);
                source.connect(scriptProcessor);
                scriptProcessor.connect(audioContext.destination);

                if (modelType === "online") {
                    dashscopeClient = new DashscopeWebSocketClient(
                        config,
                        (result) => publishCaption(result.text, result.isFinal),
                        (clientError) => {
                            console.error('Dashscope client error:', clientError);
                            setError(clientError.message);
                            setCurrentCaption(`在线识别错误：${clientError.message}`);
                        },
                    );
                    await dashscopeClient.connect();
                }

                processChunkUnlisten = await listen<{ source: string; samples: number[] }>('audio-chunk', (event) => {
                    if (event.payload.source !== "process") return;
                    queueProcessAudio(event.payload.samples);
                    processChunksSent += 1;
                });
                processStatsUnlisten = await listen<{
                    source: string;
                    chunks: number;
                    samples: number;
                    rms: number;
                    peak: number;
                    silent: boolean;
                }>('audio-capture-stats', (event) => {
                    if (event.payload.source !== "process") return;
                    const { chunks, samples, rms, peak, silent } = event.payload;
                    setProcessAudioDebug(
                        `process audio: ${silent ? 'silent' : 'signal'} | captured ${chunks} chunks / ${samples} samples | mixed ${processChunksSent} | rms ${rms.toFixed(5)} | peak ${peak.toFixed(5)}`,
                    );
                });
                processErrorUnlisten = await listen<string>('audio-transcription-error', (event) => {
                    setError(event.payload);
                    setCurrentCaption(`程序音频捕获错误：${event.payload}`);
                    void stopRecording();
                });

                setIsRecording(true);
                scriptProcessor.onaudioprocess = (event) => {
                    if (!isRecording()) return;
                    const mixed = mixWithProcessAudio(event.inputBuffer.getChannelData(0));
                    queueRecordingAudio(mixed);

                    if (modelType === "online") {
                        if (dashscopeClient?.isConnected()) dashscopeClient.sendAudio(mixed);
                        return;
                    }

                    const data = Array.from(mixed);
                    localAudioQueue = localAudioQueue.then(async () => {
                        try {
                            const result = await invoke<LocalTranscriptResult>('process_audio_chunk', { chunk: data });
                            if (result.text.trim()) publishCaption(result.text, result.isFinal);
                        } catch (inferenceError) {
                            const message = inferenceError instanceof Error ? inferenceError.message : String(inferenceError);
                            console.error('Local inference error:', inferenceError);
                            setError(message);
                            setCurrentCaption(`推理错误: ${message}`);
                        }
                    });
                };

                setProcessAudioDebug(`process audio: waiting for capture | pid ${processAudioPid}`);
                await invoke('start_process_audio_capture', { pid: processAudioPid, persistAudio: false });
                setCurrentCaption('正在同时监听麦克风和程序音频...');
                return true;
            }

            if (audioSource === "process") {
                if (!processAudioPid || processAudioPid <= 0) {
                    throw new Error("请先选择要捕获音频的 Windows 程序");
                }
                setProcessAudioDebug(`process audio: waiting for capture | pid ${processAudioPid}`);

                if (modelType === "online") {
                    const config: DashscopeConfig = {
                        apiKey: onlineApiKey || import.meta.env.VITE_DASHSCOPE_API_KEY || '',
                    };

                    if (!config.apiKey) {
                        throw new Error('在线服务访问密钥未配置');
                    }

                    dashscopeClient = new DashscopeWebSocketClient(
                        config,
                        (result) => {
                            publishCaption(result.text, result.isFinal);
                        },
                        (err) => {
                            console.error('Dashscope client error:', err);
                            setError(err.message);
                            setCurrentCaption(`在线识别错误：${err.message}`);
                        }
                    );

                    await dashscopeClient.connect();

                    processChunkUnlisten = await listen<{ source: string; samples: number[] }>('audio-chunk', (event) => {
                        if (event.payload.source === "process" && dashscopeClient?.isConnected()) {
                            dashscopeClient.sendAudio(new Float32Array(event.payload.samples));
                            processChunksSent += 1;
                            if (processChunksSent === 1) {
                                console.log("First process audio chunk sent to Dashscope.");
                            }
                        }
                    });

                    processStatsUnlisten = await listen<{
                        source: string;
                        chunks: number;
                        samples: number;
                        rms: number;
                        peak: number;
                        silent: boolean;
                    }>('audio-capture-stats', (event) => {
                        if (event.payload.source !== "process") return;

                        const { chunks, samples, rms, peak, silent } = event.payload;
                        setProcessAudioDebug(
                            `process audio: ${silent ? 'silent' : 'signal'} | captured ${chunks} chunks / ${samples} samples | sent ${processChunksSent} | rms ${rms.toFixed(5)} | peak ${peak.toFixed(5)}`,
                        );
                        if (chunks === 1) {
                            console.log("First process audio capture stats:", event.payload);
                        }
                    });

                    processErrorUnlisten = await listen<string>('audio-transcription-error', (event) => {
                        setError(event.payload);
                        setCurrentCaption(`程序音频捕获错误：${event.payload}`);
                        setIsRecording(false);
                        invoke('stop_process_audio_transcription').catch((e) => {
                            console.error("Error stopping process audio capture:", e);
                        });
                    });

                    await invoke('start_process_audio_capture', { pid: processAudioPid });
                    setIsRecording(true);
                    setCurrentCaption('正在监听程序音频...');
                    return true;
                }

                processTranscriptUnlisten = await listen<{ source: string; text: string; isFinal: boolean }>('audio-transcript', (event) => {
                    if (event.payload.source === "process" && event.payload.text.trim()) {
                        publishCaption(event.payload.text, event.payload.isFinal);
                    }
                });
                processErrorUnlisten = await listen<string>('audio-transcription-error', (event) => {
                    setError(event.payload);
                    setCurrentCaption(`程序音频捕获错误：${event.payload}`);
                    setIsRecording(false);
                    invoke('stop_process_audio_transcription').catch((e) => {
                        console.error("Error stopping process audio transcription:", e);
                    });
                });

                await invoke('start_process_audio_transcription', { pid: processAudioPid });
                setIsRecording(true);
                setCurrentCaption('正在监听程序音频...');
                return true;
            }

            // For online mode, validate credentials first before accessing microphone
            if (modelType === "online") {
                const config: DashscopeConfig = {
                    apiKey: onlineApiKey || import.meta.env.VITE_DASHSCOPE_API_KEY || '',
                };

                console.log('Checking Dashscope credentials...');
                if (!config.apiKey) {
                    throw new Error('在线服务访问密钥未配置');
                }
                console.log('Dashscope credentials validated');
            }

            // Get microphone access
            console.log('Requesting microphone access...');
            const audioConstraints: boolean | MediaTrackConstraints = audioInputDeviceId
                ? { deviceId: { exact: audioInputDeviceId } }
                : true;
            const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
            console.log('Microphone access granted');
            mediaStream = stream;

            console.log('Creating audio context...');
            const context = new AudioContext({ sampleRate: 16000 }); // Request 16kHz if possible
            console.log(`Audio context created with sample rate: ${context.sampleRate}`);
            audioContext = context;

            const source = context.createMediaStreamSource(stream);
            const processorBufferSize = modelType === "local" ? 1024 : 4096;
            const processor = context.createScriptProcessor(processorBufferSize, 1, 1);
            scriptProcessor = processor;

            source.connect(processor);
            processor.connect(context.destination);

            setIsRecording(true);
            setCurrentCaption('');

            if (modelType === "online") {
                console.log('Initializing Dashscope WebSocket client...');
                const config: DashscopeConfig = {
                    apiKey: onlineApiKey || import.meta.env.VITE_DASHSCOPE_API_KEY || '',
                };

                dashscopeClient = new DashscopeWebSocketClient(
                    config,
                    (result) => {
                        publishCaption(result.text, result.isFinal);
                    },
                    (err) => {
                        console.error('Dashscope client error:', err);
                        setError(err.message);
                        setCurrentCaption(`错误: ${err.message}`);
                    }
                );

                await dashscopeClient.connect();
                console.log('Dashscope WebSocket connected successfully');

                processor.onaudioprocess = (event) => {
                    if (!isRecording()) return;

                    const inputBuffer = event.inputBuffer.getChannelData(0);
                    queueRecordingAudio(inputBuffer);
                    if (dashscopeClient?.isConnected()) dashscopeClient.sendAudio(inputBuffer);
                };
                console.log('Online audio processing initialized');
            } else {
                // Local model inference
                console.log('Initializing local inference...');

                // Set initial caption to show local mode is active
                setCurrentCaption('本地推理模式已启动，等待音频输入...');

                processor.onaudioprocess = (event) => {
                    if (!isRecording()) return;

                    const inputBuffer = event.inputBuffer.getChannelData(0);
                    queueRecordingAudio(inputBuffer);
                    const data = Array.from(inputBuffer);

                    localAudioQueue = localAudioQueue.then(async () => {
                        try {
                            const result = await invoke<LocalTranscriptResult>('process_audio_chunk', { chunk: data });
                            if (result.text && result.text.trim().length > 0) {
                                publishCaption(result.text, result.isFinal);
                            }
                        } catch (e) {
                            console.error('Local inference error:', e);
                            const errorMsg = e instanceof Error ? e.message : String(e);
                            if (errorMsg.includes('模型未加载') || errorMsg.includes('Model not loaded')) {
                                setCurrentCaption('模型未加载，请先在设置中下载模型');
                                setError('模型未加载，请先在设置中下载模型');
                            } else {
                                setCurrentCaption(`推理错误: ${errorMsg}`);
                                setError(errorMsg);
                            }
                        }
                    });
                };
                console.log('Local audio processing initialized');
            }

            return true;

        } catch (err) {
            console.error('Failed to start recording:', err);

            // Provide more specific error messages
            let errorMessage = '启动录音失败';

            if (err instanceof Error) {
                // Check for specific error types
                if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                    errorMessage = '麦克风权限被拒绝，请允许访问麦克风';
                } else if (err.name === 'NotFoundError') {
                    errorMessage = '未找到麦克风设备';
                } else if (err.name === 'NotReadableError') {
                    errorMessage = '麦克风正在被其他应用使用';
                } else if (err.message.includes('访问密钥')) {
                    errorMessage = '在线服务访问密钥未配置，请在 Provider 中设置';
                } else if (err.message.includes('WebSocket')) {
                    errorMessage = '连接 Dashscope 服务失败，请检查网络';
                } else if (err.message.includes('not allowed')) {
                    errorMessage = 'Tauri 权限错误：' + err.message;
                } else {
                    errorMessage = err.message;
                }
            }

            setError(errorMessage);
            setCurrentCaption(errorMessage);
            if (processTranscriptUnlisten) {
                processTranscriptUnlisten();
                processTranscriptUnlisten = null;
            }
            if (processChunkUnlisten) {
                processChunkUnlisten();
                processChunkUnlisten = null;
            }
            if (processStatsUnlisten) {
                processStatsUnlisten();
                processStatsUnlisten = null;
            }
            if (processErrorUnlisten) {
                processErrorUnlisten();
                processErrorUnlisten = null;
            }
            if (isRecording()) {
                setIsTransitioning(false);
                await stopRecording();
                setIsTransitioning(true);
            }
            return false;
        } finally {
            setIsTransitioning(false);
        }
    };

    const stopRecording = async () => {
        if (isTransitioning() || !isRecording()) {
            console.log("Not recording or in transition, ignoring stop request.");
            return;
        }

        setIsTransitioning(true);
        console.log("Stopping recording...");
        setIsRecording(false);
        await flushRecordingAudio();

        if (activeAudioSource !== "microphone") {
            try {
                await invoke('stop_process_audio_transcription');
            } catch (e) {
                console.error("Error stopping process audio transcription:", e);
            }
        }

        if (activeModelType === "local") {
            try {
                await localAudioQueue;
                const finalResult = await invoke<LocalTranscriptResult>('flush_local_model');
                if (finalResult.text.trim()) publishCaption(finalResult.text, true);
            } catch (e) {
                console.error("Error flushing local transcription:", e);
            }
        }

        if (processTranscriptUnlisten) {
            processTranscriptUnlisten();
            processTranscriptUnlisten = null;
        }

        if (processChunkUnlisten) {
            processChunkUnlisten();
            processChunkUnlisten = null;
        }

        if (processStatsUnlisten) {
            processStatsUnlisten();
            processStatsUnlisten = null;
        }

        if (processErrorUnlisten) {
            processErrorUnlisten();
            processErrorUnlisten = null;
        }

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }

        if (scriptProcessor) {
            scriptProcessor.disconnect();
            scriptProcessor.onaudioprocess = null;
            scriptProcessor = null;
        }

        if (dashscopeClient) {
            await dashscopeClient.close();
            dashscopeClient = null;
        }

        if (audioContext && audioContext.state !== 'closed') {
            try {
                await audioContext.close();
            } catch (e) {
                console.error("Error closing audio context:", e);
            }
            audioContext = null;
        }

        processAudioBuffer = [];
        processAudioReadOffset = 0;

        // The caption is preserved to show the final result.
        // It will be cleared when the next recording starts.
        console.log("Recording stopped.");
        setIsTransitioning(false);
    };

    return {
        isRecording,
        isTransitioning,
        currentCaption,
        captionResult,
        error,
        processAudioDebug,
        startRecording,
        stopRecording
    };
}
