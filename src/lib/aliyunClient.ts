// Aliyun Paraformer WebSocket client for online speech recognition
// This module handles the WebSocket connection to Aliyun's real-time ASR service

export interface AliyunConfig {
    appKey: string;
    token: string;
    url?: string;
}

export interface RecognitionResult {
    text: string;
    isFinal: boolean;
    timestamp?: number;
}

export class AliyunWebSocketClient {
    private ws: WebSocket | null = null;
    private config: AliyunConfig;
    private onResult: (result: RecognitionResult) => void;
    private onError: (error: Error) => void;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 3;
    private isIntentionallyClosed = false;

    constructor(
        config: AliyunConfig,
        onResult: (result: RecognitionResult) => void,
        onError: (error: Error) => void
    ) {
        this.config = {
            ...config,
            url: config.url || 'wss://nls-gateway.cn-shanghai.aliyuncs.com/ws/v1'
        };
        this.onResult = onResult;
        this.onError = onError;
    }

    connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            try {
                this.isIntentionallyClosed = false;

                // Construct WebSocket URL with authentication
                const wsUrl = `${this.config.url}?appkey=${this.config.appKey}&token=${this.config.token}`;

                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log('Aliyun WebSocket connected');
                    this.reconnectAttempts = 0;

                    // Send start recognition command
                    this.sendStartCommand();
                    resolve();
                };

                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleMessage(data);
                    } catch (error) {
                        console.error('Failed to parse WebSocket message:', error);
                    }
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.onError(new Error('WebSocket connection error'));
                    reject(error);
                };

                this.ws.onclose = (event) => {
                    console.log('WebSocket closed:', event.code, event.reason);

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

    private sendStartCommand() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }

        const startCommand = {
            header: {
                message_id: this.generateMessageId(),
                task_id: this.generateTaskId(),
                namespace: 'SpeechTranscriber',
                name: 'StartTranscription',
                appkey: this.config.appKey
            },
            payload: {
                format: 'pcm',
                sample_rate: 16000,
                enable_intermediate_result: true,
                enable_punctuation_prediction: true,
                enable_inverse_text_normalization: true
            }
        };

        this.ws.send(JSON.stringify(startCommand));
    }

    sendAudio(audioData: Float32Array) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.warn('WebSocket not ready, skipping audio chunk');
            return;
        }

        // Convert Float32Array to Int16Array (PCM format)
        const pcmData = this.float32ToPCM(audioData);

        // Send binary audio data
        this.ws.send(pcmData.buffer);
    }

    private float32ToPCM(float32Array: Float32Array): Int16Array {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            // Clamp to [-1, 1] and convert to 16-bit PCM
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16Array;
    }

    private handleMessage(data: any) {
        const { header, payload } = data;

        if (!header || !header.name) {
            return;
        }

        switch (header.name) {
            case 'TranscriptionResultChanged':
                // Intermediate result
                if (payload && payload.result) {
                    this.onResult({
                        text: payload.result,
                        isFinal: false,
                        timestamp: Date.now()
                    });
                }
                break;

            case 'SentenceEnd':
                // Final result for a sentence
                if (payload && payload.result) {
                    this.onResult({
                        text: payload.result,
                        isFinal: true,
                        timestamp: Date.now()
                    });
                }
                break;

            case 'TranscriptionCompleted':
                console.log('Transcription completed');
                break;

            case 'TaskFailed':
                console.error('Task failed:', payload);
                this.onError(new Error(payload?.message || 'Transcription task failed'));
                break;

            default:
                console.log('Unknown message type:', header.name);
        }
    }

    stop() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }

        // Send stop command
        const stopCommand = {
            header: {
                message_id: this.generateMessageId(),
                task_id: this.generateTaskId(),
                namespace: 'SpeechTranscriber',
                name: 'StopTranscription',
                appkey: this.config.appKey
            }
        };

        this.ws.send(JSON.stringify(stopCommand));
    }

    close() {
        this.isIntentionallyClosed = true;

        if (this.ws) {
            this.stop();
            this.ws.close();
            this.ws = null;
        }
    }

    private generateMessageId(): string {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    private generateTaskId(): string {
        return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }
}
