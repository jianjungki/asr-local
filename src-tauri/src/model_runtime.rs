use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ContentPart};
use genai::resolver::{AuthData, Endpoint, ProviderConfig};
use genai::{Client, Headers, ModelIden, ServiceTarget};
use serde::Deserialize;

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModelTarget {
    pub provider: String,
    pub adapter: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ProviderTarget {
    pub adapter: String,
    pub base_url: String,
    pub api_key: Option<String>,
}

#[derive(Clone, Deserialize)]
pub(crate) struct ConversationMessage {
    pub role: String,
    pub content: String,
}

pub(crate) struct ChatInput {
    pub system_prompt: Option<String>,
    pub messages: Vec<ConversationMessage>,
    pub image_data_url: Option<String>,
}

#[derive(Clone, Default)]
pub(crate) struct GenAiModelRuntime {
    client: Client,
}

impl GenAiModelRuntime {
    pub async fn list_models(&self, target: ProviderTarget) -> Result<Vec<String>, String> {
        let adapter_name = target.adapter.trim().to_lowercase();
        let adapter = AdapterKind::from_lower_str(&adapter_name)
            .ok_or_else(|| format!("Provider 使用了不支持的适配类型：{adapter_name}"))?;
        let endpoint = Endpoint::from_owned(normalize_base_url(&target.base_url, adapter)?);
        let auth = target
            .api_key
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .map(AuthData::from_single)
            .unwrap_or(AuthData::None);
        let provider_config = ProviderConfig::default()
            .with_endpoint(endpoint)
            .with_auth(auth);

        let mut models = self
            .client
            .all_model_names(adapter, provider_config)
            .await
            .map_err(|error| format!("获取模型列表失败：{error}"))?;
        models.retain(|model| !model.trim().is_empty());
        models.sort_unstable_by_key(|model| model.to_lowercase());
        models.dedup();
        Ok(models)
    }

    pub async fn chat(&self, target: ModelTarget, input: ChatInput) -> Result<String, String> {
        let service_target = build_service_target(target)?;
        let chat_request = build_chat_request(input)?;
        let options = ChatOptions::default()
            .with_temperature(0.3)
            .with_normalize_reasoning_content(true);

        let response = self
            .client
            .exec_chat(service_target, chat_request, Some(&options))
            .await
            .map_err(|error| format!("模型调用失败：{error}"))?;

        response
            .content
            .joined_texts()
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
            .ok_or_else(|| "模型未返回文字内容".to_string())
    }
}

fn build_service_target(target: ModelTarget) -> Result<ServiceTarget, String> {
    let provider = target.provider.trim().to_lowercase();
    let adapter_name = target.adapter.trim().to_lowercase();
    let adapter = AdapterKind::from_lower_str(&adapter_name)
        .ok_or_else(|| format!("Provider 使用了不支持的适配类型：{adapter_name}"))?;
    let model = target.model.trim();
    if model.is_empty() {
        return Err("请配置要使用的文字或视觉模型".to_string());
    }

    let api_key = target
        .api_key
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let custom_without_auth = provider == "custom" && adapter == AdapterKind::OpenAI;
    if adapter != AdapterKind::Ollama && !custom_without_auth && api_key.is_none() {
        return Err("Provider 访问密钥未配置".to_string());
    }

    let base_url = normalize_base_url(&target.base_url, adapter)?;
    let auth = match api_key {
        Some(api_key) => AuthData::from_single(api_key),
        None if custom_without_auth => AuthData::RequestOverride {
            url: format!("{base_url}chat/completions"),
            headers: Headers::default(),
        },
        None => AuthData::None,
    };

    Ok(ServiceTarget {
        endpoint: Endpoint::from_owned(base_url),
        auth,
        model: ModelIden::new(adapter, model.to_string()),
    })
}

fn normalize_base_url(base_url: &str, adapter: AdapterKind) -> Result<String, String> {
    let mut base_url = base_url.trim().trim_end_matches('/');
    if base_url.is_empty() {
        return Err("Provider 服务地址未配置".to_string());
    }

    let request_suffix = if adapter == AdapterKind::Ollama {
        "/api/chat"
    } else if adapter == AdapterKind::Anthropic {
        "/messages"
    } else {
        "/chat/completions"
    };
    if let Some(value) = base_url.strip_suffix(request_suffix) {
        base_url = value.trim_end_matches('/');
    }

    Ok(format!("{base_url}/"))
}

fn build_chat_request(input: ChatInput) -> Result<ChatRequest, String> {
    let messages = validated_messages(input.messages)?;
    let last_user_index = messages.iter().rposition(|message| message.role == "user");
    let image = input
        .image_data_url
        .as_deref()
        .map(parse_image_data_url)
        .transpose()?;

    let mut request_messages = Vec::with_capacity(messages.len());
    for (index, message) in messages.into_iter().enumerate() {
        if message.role == "user" {
            if Some(index) == last_user_index {
                if let Some((content_type, base64)) = image.as_ref() {
                    request_messages.push(ChatMessage::user(vec![
                        ContentPart::from_text(message.content),
                        ContentPart::from_binary_base64(
                            content_type.clone(),
                            base64.clone(),
                            Some("desktop-capture.png".to_string()),
                        ),
                    ]));
                    continue;
                }
            }
            request_messages.push(ChatMessage::user(message.content));
        } else {
            request_messages.push(ChatMessage::assistant(message.content));
        }
    }

    let mut chat_request = ChatRequest::new(request_messages);
    if let Some(system_prompt) = input
        .system_prompt
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        chat_request = chat_request.with_system(system_prompt);
    }
    Ok(chat_request)
}

fn validated_messages(
    messages: Vec<ConversationMessage>,
) -> Result<Vec<ConversationMessage>, String> {
    let messages = messages
        .into_iter()
        .filter_map(|message| {
            let role = message.role.trim().to_lowercase();
            let content = message.content.trim().to_string();
            if content.is_empty() {
                return None;
            }
            matches!(role.as_str(), "user" | "assistant")
                .then_some(ConversationMessage { role, content })
        })
        .collect::<Vec<_>>();

    if messages.is_empty() {
        return Err("请输入要发送给模型的内容".to_string());
    }
    Ok(messages)
}

fn parse_image_data_url(value: &str) -> Result<(String, String), String> {
    let value = value.trim();
    let (metadata, data) = value
        .split_once(',')
        .ok_or_else(|| "窗口截图数据格式无效".to_string())?;
    let content_type = metadata
        .strip_prefix("data:")
        .and_then(|value| value.strip_suffix(";base64"))
        .filter(|value| value.starts_with("image/"))
        .ok_or_else(|| "窗口截图不是有效的图片数据".to_string())?;
    if data.trim().is_empty() {
        return Err("窗口截图数据为空".to_string());
    }
    Ok((content_type.to_string(), data.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use super::{
        build_chat_request, build_service_target, normalize_base_url, parse_image_data_url,
        ChatInput, ConversationMessage, ModelTarget,
    };
    use genai::adapter::AdapterKind;
    use genai::resolver::AuthData;

    #[test]
    fn normalizes_provider_base_urls() {
        assert_eq!(
            normalize_base_url("https://example.com/v1", AdapterKind::OpenAI).unwrap(),
            "https://example.com/v1/"
        );
        assert_eq!(
            normalize_base_url(
                "https://example.com/v1/chat/completions/",
                AdapterKind::OpenAI
            )
            .unwrap(),
            "https://example.com/v1/"
        );
        assert_eq!(
            normalize_base_url("http://127.0.0.1:11434/api/chat", AdapterKind::Ollama).unwrap(),
            "http://127.0.0.1:11434/"
        );
    }

    #[test]
    fn parses_image_data_urls() {
        assert_eq!(
            parse_image_data_url("data:image/png;base64,AAAA").unwrap(),
            ("image/png".to_string(), "AAAA".to_string())
        );
        assert!(parse_image_data_url("data:text/plain;base64,AAAA").is_err());
        assert!(parse_image_data_url("AAAA").is_err());
    }

    #[test]
    fn attaches_an_image_to_the_last_user_message() {
        let request = build_chat_request(ChatInput {
            system_prompt: Some("Be concise".to_string()),
            messages: vec![
                ConversationMessage {
                    role: "assistant".to_string(),
                    content: "Ready".to_string(),
                },
                ConversationMessage {
                    role: "user".to_string(),
                    content: "What is visible?".to_string(),
                },
            ],
            image_data_url: Some("data:image/png;base64,AAAA".to_string()),
        })
        .unwrap();

        assert_eq!(request.system.as_deref(), Some("Be concise"));
        assert_eq!(request.messages.len(), 2);
        assert_eq!(
            request.messages[1].content.first_text(),
            Some("What is visible?")
        );
        let images = request.messages[1].content.binaries();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].content_type, "image/png");
    }

    #[test]
    fn permits_an_unauthenticated_custom_openai_endpoint() {
        let target = build_service_target(ModelTarget {
            provider: "custom".to_string(),
            adapter: "openai".to_string(),
            base_url: "http://127.0.0.1:8080/v1".to_string(),
            api_key: None,
            model: "local-model".to_string(),
        })
        .unwrap();

        match target.auth {
            AuthData::RequestOverride { url, headers } => {
                assert_eq!(url, "http://127.0.0.1:8080/v1/chat/completions");
                assert_eq!(headers.iter().count(), 0);
            }
            _ => panic!("custom endpoint should use a request override"),
        }
    }
}
