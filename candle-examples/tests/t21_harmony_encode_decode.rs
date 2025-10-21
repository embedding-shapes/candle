use anyhow::Result;
use openai_harmony::{
    chat::{Message, Role},
    load_harmony_encoding, HarmonyEncodingName,
};

// T21 Harmony encode/decode: round-trip for the example’s prompt format.
// Validates that rendering the conversation (user prompt -> completion tokens)
// decodes to a string which, when re-encoded with allowed special tokens,
// produces the exact same token sequence. Also checks decoded structure.
#[test]
fn t21_harmony_encode_decode_roundtrip() -> Result<()> {
    let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;

    // Example prompt used in the example binary.
    let prompt = "Explain what MXFP4 quantization is".to_string();
    let user_msg = Message::from_role_and_content(Role::User, prompt.clone());

    // Render tokens for completion (adds the next assistant header).
    let tokens = enc.render_conversation_for_completion([&user_msg], Role::Assistant, None)?;

    // Decode via Harmony tokenizer.
    let decoded = enc.tokenizer().decode_utf8(tokens.iter().copied())?;

    // Re-encode with all special tokens allowed and ensure perfect round-trip.
    let allowed = enc.tokenizer().special_tokens();
    let (tokens_rt, _unstable) = enc.tokenizer().encode(&decoded, &allowed);
    assert_eq!(tokens, tokens_rt, "round-trip tokens should match exactly");

    // The decoded string should contain the user header and the next assistant header.
    assert!(decoded.starts_with("<|start|>user<|message|>"));
    assert!(decoded.ends_with("<|start|>assistant"));

    // As a lightweight structural check, ensure the prompt text appears once
    // in the decoded string and is placed between <|message|> and <|end|>.
    let msg_pos = decoded.find("<|message|>").expect("missing <|message|>") + "<|message|>".len();
    let end_pos = decoded.rfind("<|end|>").expect("missing <|end|>");
    let inner = &decoded[msg_pos..end_pos];
    assert!(inner.contains(&prompt));

    Ok(())
}
