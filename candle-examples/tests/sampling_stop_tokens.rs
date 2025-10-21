use openai_harmony::{load_harmony_encoding, HarmonyEncodingName};

#[test]
fn sampling_stop_tokens_match_harmony_assistant_actions() -> Result<(), Box<dyn std::error::Error>>
{
    let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let tok = enc.tokenizer();
    let id = |s: &str| -> u32 {
        let vs = tok.encode_with_special_tokens(s);
        assert_eq!(vs.len(), 1, "expected single id for {s}");
        vs[0]
    };
    let id_end = id("<|end|>");
    let id_return = id("<|return|>");
    let id_call = id("<|call|>");

    let stop_actions = enc.stop_tokens_for_assistant_actions()?;
    assert!(
        stop_actions.contains(&id_return),
        "<|return|> must be a stop token"
    );
    assert!(
        stop_actions.contains(&id_call),
        "<|call|> must be a stop token"
    );
    assert!(
        !stop_actions.contains(&id_end),
        "<|end|> must not be a stop in sampling loop"
    );
    Ok(())
}
