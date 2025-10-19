use anyhow::Result;
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName};

// T22 Harmony special token ids: ensure expected numeric ids for key tokens used in generation.
#[test]
fn t22_harmony_special_token_ids() -> Result<()> {
    let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let tok = enc.tokenizer();

    // Helper to encode a single special token and return its id
    let eid = |s: &str| -> anyhow::Result<u32> {
        let ids = tok.encode_with_special_tokens(s);
        if ids.len() != 1 {
            anyhow::bail!("expected exactly one id for {s}, got {:?}", ids);
        }
        Ok(ids[0])
    };

    let id_message = eid("<|message|>")?;
    let id_channel = eid("<|channel|>")?;
    let id_end = eid("<|end|>")?;
    let id_call = eid("<|call|>")?;
    let id_return = eid("<|return|>")?;

    assert_eq!(id_message, 200008);
    assert_eq!(id_channel, 200005);
    assert_eq!(id_end, 200007); // EOS
    assert_eq!(id_call, 200012); // also an EOS for tools
    assert_eq!(id_return, 200002); // primary EOS

    Ok(())
}
