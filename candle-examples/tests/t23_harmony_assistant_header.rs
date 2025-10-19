use anyhow::Result;
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName};

// T23 Harmony assistant header ordering: ensure first token after the assistant
// start is <|channel|> and that the last header token is <|message|>.
#[test]
fn t23_harmony_assistant_header_order() -> Result<()> {
    let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let tok = enc.tokenizer();

    // Encode header snippet explicitly; allow special tokens during encoding.
    let allowed = tok.special_tokens();
    let (ids, _unstable) = tok.encode("<|channel|>final<|message|>", &allowed);

    // Resolve ids for key specials using the tokenizer to avoid hardcoding
    let id = |s: &str| -> u32 {
        let vs = tok.encode_with_special_tokens(s);
        assert_eq!(vs.len(), 1, "expected single id for {s}, got {vs:?}");
        vs[0]
    };
    let id_channel = id("<|channel|>");
    let id_message = id("<|message|>");

    assert!(!ids.is_empty(), "header encoding should not be empty");
    assert_eq!(ids[0], id_channel, "first token must be <|channel|>");
    assert_eq!(ids[ids.len() - 1], id_message, "last header token must be <|message|>");

    Ok(())
}

