// Dashscope WebSocket client for real-time speech recognition (new protocol)
// This module handles the WebSocket connection to Dashscope's real-time ASR service.

export interface DashscopeConfig {
    apiKey: string;
    model?: string;
    url?: string;
}

export interface RecognitionResult {
    text: string;
    isFinal: boolean;
    timestamp?: number;
}

export class DashscopeWebSocketClient {
    private ws: WebSocket | null = null;
    private config: DashscopeConfig;
    private onResult: (result: RecognitionResult) => void;
    private onError: (error: Error) => void;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 3;
    private isIntentionallyClosed = false;
    private taskStarted = false;
    private taskId: string;

    constructor(
        config: DashscopeConfig,
        onResult: (result: RecognitionResult) => void,
        onError: (error: Error) => void
    ) {
        this.config = {
            ...config,
            model: config.model || 'paraformer-realtime-v2',//'fun-asr-realtime',
            url: config.url || 'wss://dashscope.aliyuncs.com/api-ws/v1/inference/'
        };
        this.onResult = onResult;
        this.onError = onError;
        this.taskId = this.generateTaskId();
    }

    connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            try {
                this.isIntentionallyClosed = false;
                this.taskStarted = false;

                // For browser, API key is passed as a URL parameter
                const wsUrl = `${this.config.url}?api_key=${this.config.apiKey}`;

                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log('Dashscope WebSocket connected');
                    this.reconnectAttempts = 0;
                    this.sendRunTask();
                    resolve();
                };

                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data as string);
                        this.handleMessage(data);
                    } catch (error) {
                        console.error('Failed to parse WebSocket message:', error);
                    }
                };

                this.ws.onerror = (error) => {
                    if (this.isIntentionallyClosed) return;
                    console.error('WebSocket error:', error);
                    this.onError(new Error('WebSocket connection error'));
                    reject(error);
                };

                this.ws.onclose = (event) => {
                    console.log('WebSocket closed:', event.code, event.reason);
                    if (!this.taskStarted && !this.isIntentionallyClosed) {
                        this.onError(new Error('Task failed to start, connection closed.'));
                    }

                    if (!this.isIntentionallyClosed && this.reconnectAttempts < this.maxReconnectAttempts) {
                        this.reconnectAttempts++;
                        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
                        setTimeout(() => this.connect(), 1000 * this.reconnectAttempts);
                    }
                };
            } catch (error) {
                reject(error);
            }
        });
    }

    private sendRunTask() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        const runTaskMessage = {
            header: {
                action: 'run-task',
                task_id: this.taskId,
                streaming: 'duplex'
            },
            payload: {
                task_group: 'audio',
                task: 'asr',
                function: 'recognition',
                model: this.config.model,
                parameters: {
                    sample_rate: 16000,
                    format: 'pcm' // We are sending raw PCM data
                },
                input: {}
            }
        };
        this.ws.send(JSON.stringify(runTaskMessage));
        console.log('Sent run-task command.');
    }

    sendAudio(audioData: Float32Array) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.taskStarted) {
            // console.warn('WebSocket not ready or task not started, skipping audio chunk');
            return;
        }
        const pcmData = this.float32ToPCM(audioData);
        this.ws.send(pcmData.buffer);
    }

    private float32ToPCM(float32Array: Float32Array): Int16Array {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16Array;
    }

    private handleMessage(data: any) {
        const { header, payload } = data;
        if (!header || !header.event) return;

        switch (header.event) {
            case 'task-started':
                console.log('Task started.');
                this.taskStarted = true;
                break;
            case 'result-generated':
                const sentence = payload?.output?.sentence;
                if (sentence) {
                    const text = sentence.result || sentence.text;
                    if (text) {
                        this.onResult({
                            text: text,
                            isFinal: sentence.sentence_end === true || typeof sentence.end_time === 'number',
                            timestamp: Date.now()
                        });
                    }
                }
                break;
            case 'task-finished':
                console.log('Task finished.');
                this.close();
                break;
            case 'task-failed':
                console.error('Task failed:', header.error_message);
                this.onError(new Error(header.error_message || 'Transcription task failed'));
                this.close();
                break;
            default:
                console.log('Unknown event:', header.event);
        }
    }

    private sendFinishTask() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        const finishTaskMessage = {
            header: {
                action: 'finish-task',
                task_id: this.taskId,
                streaming: 'duplex'
            },
            payload: {
                input: {}
            }
        };
        this.ws.send(JSON.stringify(finishTaskMessage));
        console.log('Sent finish-task command.');
    }

    close() {
        this.isIntentionallyClosed = true;
        if (this.ws) {
            if (this.ws.readyState === WebSocket.OPEN && this.taskStarted) {
                this.sendFinishTask();
            }
            // Give a moment for the finish-task message to be sent before closing
            setTimeout(() => {
                if (this.ws) {
                    this.ws.close();
                    this.ws = null;
                }
            }, 100);
        }
        this.taskStarted = false;
    }

    private generateTaskId(): string {
        // Basic UUID v4 generator
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
            const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN && this.taskStarted;
    }
}
