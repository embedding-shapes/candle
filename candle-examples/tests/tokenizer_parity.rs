use anyhow::Result;
use gpt_oss_tokenizer::{render_then_encode, load_stop_token_ids, lookup_special_ids, load_special_ids_from_map};
use openai_harmony::chat::{Message, Role};

const SNAPSHOT_DIR: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";

fn p(s: &str) -> std::path::PathBuf {
    if let Some(rest) = s.strip_prefix("~/") {
        std::env::var("HOME").map(std::path::PathBuf::from).unwrap().join(rest)
    } else {
        std::path::PathBuf::from(s)
    }
}

#[test]
fn t30_tokenizer_parity_three_cases() -> Result<()> {
    let dir = p(SNAPSHOT_DIR);

    // Base fixtures path is at workspace root tests/fixtures
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures");

    // Case 1: simple one-turn
    let ids_ref: Vec<u32> = {
        let f = std::fs::File::open(fixtures.join("chat_simple.json"))?;
        let v: serde_json::Value = serde_json::from_reader(f)?;
        v.get("input_ids").unwrap().as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
    };
    let tokens = render_then_encode(&dir, &[Message::from_role_and_content(Role::User, "Explain what MXFP4 quantization is")], true)?;
    assert_eq!(tokens, ids_ref, "simple parity failed");

    // Case 2: system+dev+user
    let ids_ref: Vec<u32> = {
        let f = std::fs::File::open(fixtures.join("chat_system_dev.json"))?;
        let v: serde_json::Value = serde_json::from_reader(f)?;
        v.get("input_ids").unwrap().as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
    };
    let msgs = vec![
        Message::from_role_and_content(Role::System, "You are a helpful assistant."),
        Message::from_role_and_content(Role::Developer, "Answer concisely."),
        Message::from_role_and_content(Role::User, "Who founded OpenAI?"),
    ];
    let tokens = render_then_encode(&dir, &msgs, true)?;
    assert_eq!(tokens, ids_ref, "system/dev parity failed");

    // Case 3: two-turn (user-assistant-user)
    let ids_ref: Vec<u32> = {
        let f = std::fs::File::open(fixtures.join("chat_two_turns.json"))?;
        let v: serde_json::Value = serde_json::from_reader(f)?;
        v.get("input_ids").unwrap().as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
    };
    let msgs = vec![
        Message::from_role_and_content(Role::User, "Hi"),
        Message::from_role_and_content(Role::Assistant, "Hello"),
        Message::from_role_and_content(Role::User, "What is YARN RoPE?"),
    ];
    let tokens = render_then_encode(&dir, &msgs, true)?;
    assert_eq!(tokens, ids_ref, "two-turn parity failed");

    Ok(())
}

#[test]
fn t31_special_ids_match_generation_config() -> Result<()> {
    let dir = p(SNAPSHOT_DIR);
    let (bos, pad_tok, ret, call) = lookup_special_ids(&dir)?;
    // Load stop ids (eos list + pad) from generation config
    let stops = load_stop_token_ids(&dir)?;
    // Expectations from HF config: BOS=199998, PAD=199999, EOS contains <|return|>, <|call|>.
    assert_eq!(bos, 199998);
    assert_eq!(pad_tok, 199999);
    assert!(stops.contains(&ret), "<|return|> not in stop set");
    assert!(stops.contains(&call), "<|call|> not in stop set");
    assert!(stops.contains(&pad_tok), "pad not in stop set");
    Ok(())
}

#[test]
fn t32_special_tokens_map_validation() -> Result<()> {
    let dir = p(SNAPSHOT_DIR);
    // Map the BOS/EOS/PAD token strings from special_tokens_map.json and confirm expected ids.
    let (bos, pad, eos) = load_special_ids_from_map(&dir)?;
    assert_eq!(bos, 199998, "bos id mismatch vs expectation");
    assert_eq!(pad, 199999, "pad id mismatch vs expectation");
    assert_eq!(eos, 200002, "eos(<|return|>) id mismatch vs expectation");
    Ok(())
}
