use anyhow::Result;
use openai_harmony::{
    chat::{Message, Role},
    load_harmony_encoding, HarmonyEncodingName,
};

// T23 Harmony parse: ensure assistant header requires <|channel|> then text under <|message|>,
// and that parsing yields the expected assistant message with channel "final".
#[test]
fn t23_harmony_parse_messages_with_channel() -> Result<()> {
    let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let tok = enc.tokenizer();

    // Build a minimal conversation with one user message.
    let user_msg = Message::from_role_and_content(Role::User, "hi".to_string());
    let mut tokens = enc.render_conversation_for_completion([&user_msg], Role::Assistant, None)?;

    // Prefill assistant header: <|channel|>final<|message|>
    let allowed = tok.special_tokens();
    let (hdr, _unstable) = tok.encode("<|channel|>final<|message|>", &allowed);
    tokens.extend(hdr);

    // Add small content and then terminate with <|return|> (do not use <|end|> as a decode stop).
    tokens.extend(tok.encode_ordinary("hello"));
    let (trailer, _unstable) = tok.encode("<|return|>", &allowed);
    tokens.extend(trailer);

    // Parse back to messages.
    let messages = enc.parse_messages_from_completion_tokens(tokens.iter().copied(), None)?;
    assert!(!messages.is_empty(), "no messages parsed");
    let last = messages.last().unwrap();
    assert_eq!(last.author.role, Role::Assistant);
    assert_eq!(last.channel.as_deref(), Some("final"));
    let mut content = String::new();
    for c in &last.content {
        if let openai_harmony::chat::Content::Text(t) = c {
            content.push_str(&t.text);
        }
    }
    assert!(content.contains("hello"));
    Ok(())
}
