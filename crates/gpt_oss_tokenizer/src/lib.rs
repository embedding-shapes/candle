use anyhow::{Context as _, Result};
use minijinja::{context, Environment};
use openai_harmony::chat::{Message as HarmonyMessage, Role as HarmonyRole, Content};
use serde::Serialize;
use std::path::Path;
use tokenizers::Tokenizer;

// Constants / defaults
const CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";
const TOKENIZER_FILE: &str = "tokenizer.json";
const GENERATION_CONFIG_FILE: &str = "generation_config.json";
const SPECIAL_TOKENS_MAP_FILE: &str = "special_tokens_map.json";

// Harmony special token markers used in decoded text parsing
const ST_START: &str = "<|start|>";
const ST_MESSAGE: &str = "<|message|>";
const ST_END: &str = "<|end|>";
const ST_CALL: &str = "<|call|>";
const ST_RETURN: &str = "<|return|>";
const ST_CHANNEL: &str = "<|channel|>";
const ST_CONSTRAIN: &str = "<|constrain|>";

#[derive(Debug, Clone, Serialize)]
struct JinjaMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
}

fn map_role(role: &HarmonyRole) -> &'static str {
    match role {
        HarmonyRole::System => "system",
        HarmonyRole::Developer => "developer",
        HarmonyRole::User => "user",
        HarmonyRole::Assistant => "assistant",
        // Default conservative mapping
        _ => "user",
    }
}

/// Render the chat template from the model snapshot and encode the resulting prompt text
/// using the tokenizer.json from the same snapshot. Returns token ids.
pub fn render_then_encode<P: AsRef<Path>>(
    snapshot_dir: P,
    messages: &[HarmonyMessage],
    add_generation_prompt: bool,
) -> Result<Vec<u32>> {
    let snapshot_dir = snapshot_dir.as_ref();
    let template_path = snapshot_dir.join(CHAT_TEMPLATE_FILE);
    let tokenizer_path = snapshot_dir.join(TOKENIZER_FILE);

    let template_src = std::fs::read_to_string(&template_path)
        .with_context(|| format!("failed to read {}", template_path.display()))?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;

    // Map harmony messages → jinja messages as expected by HF template
    let mut jmsgs = Vec::with_capacity(messages.len());
    for m in messages {
        let role = map_role(&m.author.role).to_string();
        // Flatten message content (concatenate textual segments)
        let mut text = String::new();
        for c in &m.content {
            match c {
                Content::Text(tc) => {
                    if !text.is_empty() { text.push_str(""); }
                    text.push_str(&tc.text);
                }
                Content::SystemContent(sc) => {
                    // Serialize structured system content as JSON fallback.
                    if !text.is_empty() { text.push_str(""); }
                    let s = serde_json::to_string(sc).unwrap_or_default();
                    text.push_str(&s);
                }
                Content::DeveloperContent(dc) => {
                    if !text.is_empty() { text.push_str(""); }
                    let s = serde_json::to_string(dc).unwrap_or_default();
                    text.push_str(&s);
                }
            }
        }
        let content = Some(text);
        let thinking = None;
        jmsgs.push(JinjaMessage { role, content, thinking });
    }

    // Render template using minijinja
    let mut env = Environment::new();
    // Provide strftime_now used by the HF template.
    env.add_function("strftime_now", |fmt: String| {
        let now = chrono::Local::now();
        now.format(&fmt).to_string()
    });
    env.add_template("chat_template", &template_src)?;
    let tmpl = env.get_template("chat_template")?;
    let rendered = tmpl.render(context! { messages => jmsgs, add_generation_prompt => add_generation_prompt })
        .context("failed to render chat template")?;

    // Encode with tokenizer; special tokens appear verbatim in the rendered string
    // and are registered as added tokens in tokenizer.json, so a regular encode preserves them.
    let enc = tokenizer
        .encode(rendered, /*add_special_tokens=*/false)
        .map_err(|e| anyhow::anyhow!("tokenizer.encode failed: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

/// Load stop token ids from the model's generation_config.json.
/// Specifically, includes: eos_token_id(s) and pad_token_id.
pub fn load_stop_token_ids<P: AsRef<Path>>(snapshot_dir: P) -> Result<Vec<u32>> {
    #[derive(serde::Deserialize)]
    struct GenCfg {
        #[serde(default)]
        eos_token_id: Option<serde_json::Value>,
        #[serde(default)]
        pad_token_id: Option<u32>,
    }
    let path = snapshot_dir.as_ref().join(GENERATION_CONFIG_FILE);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let cfg: GenCfg = serde_json::from_slice(&bytes).context("invalid generation_config.json")?;

    let mut ids = Vec::new();
    if let Some(eos) = cfg.eos_token_id {
        match eos {
            serde_json::Value::Number(n) => {
                if let Some(v) = n.as_u64() { ids.push(v as u32) }
            }
            serde_json::Value::Array(arr) => {
                for v in arr {
                    if let Some(n) = v.as_u64() { ids.push(n as u32) }
                }
            }
            _ => {}
        }
    }
    if let Some(pad) = cfg.pad_token_id { ids.push(pad) }
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

/// Quick helper for verifying special token ids for GPT-OSS.
pub fn lookup_special_ids<P: AsRef<Path>>(snapshot_dir: P) -> Result<(u32, u32, u32, u32)> {
    let tokenizer_path = snapshot_dir.as_ref().join(TOKENIZER_FILE);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;
    let bos = tokenizer.token_to_id("<|startoftext|>").context("missing BOS token id")?;
    let pad = tokenizer.token_to_id("<|endoftext|>").context("missing PAD token id")?;
    let ret = tokenizer.token_to_id("<|return|>").context("missing <|return|> id")?;
    let call = tokenizer.token_to_id("<|call|>").context("missing <|call|> id")?;
    Ok((bos, pad, ret, call))
}

/// Load special token ids as declared in special_tokens_map.json by mapping their string
/// representations through the tokenizer.json. This validates that BOS/EOS/PAD are
/// consistent between the map and the tokenizer ids.
pub fn load_special_ids_from_map<P: AsRef<Path>>(snapshot_dir: P) -> Result<(u32, u32, u32)> {
    #[derive(serde::Deserialize)]
    struct MapCfg {
        bos_token: String,
        eos_token: serde_json::Value,
        pad_token: String,
    }
    let base = snapshot_dir.as_ref();
    let map_path = base.join(SPECIAL_TOKENS_MAP_FILE);
    let tok_path = base.join(TOKENIZER_FILE);
    let bytes = std::fs::read(&map_path)
        .with_context(|| format!("failed to read {}", map_path.display()))?;
    let cfg: MapCfg = serde_json::from_slice(&bytes).context("invalid special_tokens_map.json")?;
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;

    let bos = tokenizer
        .token_to_id(&cfg.bos_token)
        .with_context(|| format!("missing '{}' in tokenizer", cfg.bos_token))?;
    let pad = tokenizer
        .token_to_id(&cfg.pad_token)
        .with_context(|| format!("missing '{}' in tokenizer", cfg.pad_token))?;
    let eos = match cfg.eos_token {
        serde_json::Value::String(s) => tokenizer
            .token_to_id(&s)
            .with_context(|| format!("missing '{}' in tokenizer", s))?,
        serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
            // HF sometimes stores multi-eos in generation_config, not in special_tokens_map.
            // Here we only support string eos in the map for validation purposes.
            anyhow::bail!("unsupported eos_token form in special_tokens_map.json, expected string")
        }
        _ => anyhow::bail!("invalid eos_token in special_tokens_map.json"),
    };
    Ok((bos, pad, eos))
}

/// Extract the assistant's final channel message content from a decoded Harmony string.
/// Returns the inner text between the assistant header `<|channel|>final<|message|>`
/// and the next control token (one of `<|return|>`, `<|call|>`, `<|end|>`, or the
/// start of another header `<|start|>`). Control tokens are never included.
pub fn extract_final_assistant_text_from_decoded(decoded: &str) -> Option<String> {
    // Prefer an explicit match on the 'final' channel, if present.
    let preferred = format!("{}assistant{}final{}", ST_START, ST_CHANNEL, ST_MESSAGE);
    let slice_after = if let Some(pos) = decoded.rfind(&preferred) {
        &decoded[pos + preferred.len()..]
    } else {
        // Fallback: find the last assistant header, then locate the next message marker.
        let start_tag = format!("{}assistant", ST_START);
        let start_pos = decoded.rfind(&start_tag)?;
        let rest = &decoded[start_pos + start_tag.len()..];
        if let Some(pos) = rest.find(ST_CHANNEL) {
            let rest2 = &rest[pos + ST_CHANNEL.len()..];
            let mpos = rest2.find(ST_MESSAGE)?;
            &rest2[mpos + ST_MESSAGE.len()..]
        } else {
            let mpos = rest.find(ST_MESSAGE)?;
            &rest[mpos + ST_MESSAGE.len()..]
        }
    };

    let mut end_idx = slice_after.len();
    for stop in [ST_RETURN, ST_CALL, ST_END, ST_START] {
        if let Some(p) = slice_after.find(stop) {
            end_idx = end_idx.min(p);
        }
    }
    Some(slice_after[..end_idx].to_string())
}

/// Compute which Harmony special tokens must remain allowed (not masked) for the
/// very next token emission, given the decoded Harmony string observed so far.
///
/// This reflects the minimal grammar used during completion:
/// - After "...<|start|>assistant" → must allow <|channel|>.
/// - After "...<|channel|>" → allow free-form channel text (e.g. "final|analysis|commentary"),
///   then require <|message|> as the next special control.
/// - After "...<|message|>..." → allow free text plus terminators <|end|>, <|return|>, <|call|>;
///   never pre-zero their logits; stopping is handled on emission.
///
/// Returns the set of special token strings that MUST NOT be masked at this step.
pub fn allowed_specials_for_next(decoded_so_far: &str) -> std::collections::BTreeSet<&'static str> {
    let mut allow: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();

    // Find the last assistant header start, if any.
    let assistant_hdr = format!("{}assistant", ST_START);
    let last_asst = decoded_so_far.rfind(&assistant_hdr);
    if last_asst.is_none() {
        // No assistant yet: do not enforce any special allowances.
        return allow;
    }
    let after_asst = &decoded_so_far[last_asst.unwrap() + assistant_hdr.len()..];

    // Case 1: immediately after assistant header, before <|channel|>
    if !after_asst.contains(ST_CHANNEL) {
        allow.insert(ST_CHANNEL);
        return allow;
    }

    // Strip up to and including <|channel|>
    let ch_pos = after_asst.rfind(ST_CHANNEL).unwrap();
    let after_channel = &after_asst[ch_pos + ST_CHANNEL.len()..];
    // If <|message|> has not yet appeared after channel, permit only the <|message|> control
    // (channel qualifier like "final"/"analysis"/"commentary" are normal tokens, not specials).
    if !after_channel.contains(ST_MESSAGE) {
        allow.insert(ST_MESSAGE);
        return allow;
    }

    // After <|message|>, allow terminators but do not force-stop via masking.
    allow.insert(ST_END);
    allow.insert(ST_RETURN);
    allow.insert(ST_CALL);
    // Also keep <|constrain|> unmasked if present in this vocabulary.
    allow.insert(ST_CONSTRAIN);
    allow
}

/// Apply a conservative mask to logits that would otherwise suppress Harmony specials
/// globally: ensure the required specials for the next step remain untouched.
///
/// - `logits` is a mutable slice of length equal to the vocabulary size.
/// - `tokenizer` maps special token strings to ids.
/// - `decoded_so_far` is the full decoded Harmony string up to (and including) the last token.
/// - `globally_forbidden_special_ids` is an optional set of ids that external code intends to
///   suppress; this function will un-suppress the specials required by the Harmony grammar.
pub fn unmask_required_harmony_specials(
    logits: &mut [f32],
    tokenizer: &Tokenizer,
    decoded_so_far: &str,
    globally_forbidden_special_ids: &std::collections::BTreeSet<u32>,
) {
    let allow = allowed_specials_for_next(decoded_so_far);
    for s in allow {
        if let Some(id) = tokenizer.token_to_id(s) {
            if globally_forbidden_special_ids.contains(&id) {
                // Ensure this id remains usable by restoring logits to finite range if needed.
                if logits.get_mut(id as usize).is_some() {
                    let v = &mut logits[id as usize];
                    if !v.is_finite() || *v <= f32::NEG_INFINITY / 2.0 {
                        // Reset to a neutral small logit instead of -inf; caller may overwrite.
                        *v = 0.0;
                    }
                }
            }
        }
    }
}
