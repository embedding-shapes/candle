use anyhow::Result;
use openai_harmony::{chat::{Message, Role}, load_harmony_encoding, HarmonyEncodingName};
use gpt_oss_tokenizer::extract_final_assistant_text_from_decoded;

// Feed a minimal prompt that yields both an analysis channel and a final channel
// and assert our extractor returns exactly the same bytes as Harmony's parsed
// final message content.
#[test]
fn harmony_extract_final_matches_harmony_parser() -> Result<()> {
    let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let tok = enc.tokenizer();

    // User message
    let user_msg = Message::from_role_and_content(Role::User, "hi".to_string());

    // Start directly at the assistant completion start as Harmony renders it.
    let mut toks: Vec<u32> = enc
        .render_conversation_for_completion([&user_msg], Role::Assistant, None)?
        .into_iter()
        .collect();

    // First assistant message: analysis channel
    let allowed = tok.special_tokens();
    let (hdr_a, _unstable) = tok.encode("<|channel|>analysis<|message|>", &allowed);
    toks.extend(hdr_a);
    toks.extend(tok.encode_ordinary("thinking"));
    let (end, _unstable) = tok.encode("<|end|>", &allowed);
    toks.extend(end);

    // Second assistant message: final channel with return terminator
    // No explicit newline boundary: Harmony parser expects the next start token directly.
    toks.extend(tok.encode_with_special_tokens("<|start|>"));
    toks.extend(tok.encode_ordinary("assistant"));
    let (hdr_f, _unstable) = tok.encode("<|channel|>final<|message|>", &allowed);
    toks.extend(hdr_f);
    let expected = "the answer.".to_string();
    toks.extend(tok.encode_ordinary(&expected));
    let (ret, _unstable) = tok.encode("<|return|>", &allowed);
    toks.extend(ret);

    // Decode and extract using our helper
    let decoded = tok.decode_utf8(toks.iter().copied())?;
    let extracted = extract_final_assistant_text_from_decoded(&decoded)
        .expect("failed to extract final channel message");

    // Ground truth: parse messages via Harmony and pick the last assistant/final
    let messages = enc.parse_messages_from_completion_tokens(toks.into_iter(), None)?;
    let last = messages.last().expect("no parsed messages");
    assert_eq!(last.author.role, Role::Assistant);
    assert_eq!(last.channel.as_deref(), Some("final"));
    let mut final_text = String::new();
    for c in &last.content {
        if let openai_harmony::chat::Content::Text(t) = c { final_text.push_str(&t.text); }
    }

    assert_eq!(extracted, final_text, "extracted bytes must match Harmony parsing exactly");
    assert_eq!(extracted, expected, "sanity: content should equal the inserted final text");
    Ok(())
}
