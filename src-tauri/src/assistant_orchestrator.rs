use crate::model_runtime::{ChatInput, ConversationMessage, GenAiModelRuntime, ModelTarget};
use serde::Deserialize;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AssistantModelSet {
    pub controller: ModelTarget,
    pub vision: Option<ModelTarget>,
    #[allow(dead_code)] // Reserved for a Rig speech tool/model binding.
    pub speech: Option<ModelTarget>,
}

pub(crate) struct AssistantRequest {
    pub models: AssistantModelSet,
    pub system_prompt: Option<String>,
    pub messages: Vec<ConversationMessage>,
    pub image_data_url: Option<String>,
    #[allow(dead_code)] // Direct mode never performs desktop actions.
    pub allow_desktop_actions: bool,
}

pub(crate) type OrchestratorFuture<'a> =
    Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>>;

/// Stable boundary for the assistant execution strategy. A future Rig-backed
/// implementation can use controller, vision, speech, and desktop tools while
/// keeping the Tauri command and Provider settings unchanged.
pub(crate) trait AssistantOrchestrator: Send + Sync {
    fn respond(&self, request: AssistantRequest) -> OrchestratorFuture<'_>;
}

impl AssistantModelSet {
    fn into_direct_target(self, needs_vision: bool) -> Result<ModelTarget, String> {
        if needs_vision {
            self.vision
                .ok_or_else(|| "视觉模型 Provider 未配置".to_string())
        } else {
            Ok(self.controller)
        }
    }
}

#[derive(Default)]
struct DirectAssistantOrchestrator {
    runtime: GenAiModelRuntime,
}

impl AssistantOrchestrator for DirectAssistantOrchestrator {
    fn respond(&self, request: AssistantRequest) -> OrchestratorFuture<'_> {
        Box::pin(async move {
            let AssistantRequest {
                models,
                system_prompt,
                messages,
                image_data_url,
                allow_desktop_actions: _,
            } = request;

            let target = models.into_direct_target(image_data_url.is_some())?;

            self.runtime
                .chat(
                    target,
                    ChatInput {
                        system_prompt,
                        messages,
                        image_data_url,
                    },
                )
                .await
        })
    }
}

pub(crate) struct AssistantState {
    orchestrator: Arc<dyn AssistantOrchestrator>,
}

impl Default for AssistantState {
    fn default() -> Self {
        Self {
            orchestrator: Arc::new(DirectAssistantOrchestrator::default()),
        }
    }
}

impl AssistantState {
    pub async fn respond(&self, request: AssistantRequest) -> Result<String, String> {
        self.orchestrator.respond(request).await
    }
}
