use std::fs;
use std::io::Write;
use std::path::PathBuf;

// Keep imports local to avoid unused warnings when running partial test suites.
use gpt_oss_tokenizer::{extract_final_assistant_text_from_decoded, load_stop_token_ids};

fn write_file(dir: &std::path::Path, name: &str, body: &str) {
    let p = dir.join(name);
    let mut f = fs::File::create(&p).expect("create file");
    f.write_all(body.as_bytes()).expect("write file");
}

#[test]
fn t01_extract_final_text_prefers_final_channel_and_strips_controls() {
    // Construct a decoded Harmony string containing multiple turns and control tokens.
    let decoded = "<|start|>system<|message|>You are helpful<|end|>\
                   <|start|>user<|message|>Hi!<|end|>\
                   <|start|>assistant<|channel|>draft<|message|>Interim<|end|>\
                   <|start|>assistant<|channel|>final<|message|>Hello, world!<|return|>{}";

    let out = extract_final_assistant_text_from_decoded(decoded).expect("some text");
    assert_eq!(out, "Hello, world!");
}

#[test]
fn t02_extract_last_assistant_when_no_final_channel() {
    // No explicit final channel: fall back to last assistant header + message section.
    let decoded = "<|start|>assistant<|message|>First\n<|end|>\
                   <|start|>assistant<|message|>Second answer<|call|>tool";
    let out = extract_final_assistant_text_from_decoded(decoded).expect("some text");
    assert_eq!(out, "Second answer");
}

#[test]
fn t03_extract_stops_before_any_control_token() {
    // Ensure we stop at any of the control markers (<|return|>, <|call|>, <|end|>, <|start|>).
    let decoded = "<|start|>assistant<|channel|>final<|message|>X Y Z<|end|>garbage";
    let out = extract_final_assistant_text_from_decoded(decoded).expect("some text");
    assert_eq!(out, "X Y Z");

    let decoded2 = "<|start|>assistant<|channel|>final<|message|>A B<|call|>fn";
    let out2 = extract_final_assistant_text_from_decoded(decoded2).expect("some text");
    assert_eq!(out2, "A B");
}

#[test]
fn t10_load_stop_token_ids_handles_scalar_and_array_and_dedups() {
    // Prepare a temporary snapshot dir with generation_config.json containing both forms.
    // Create a unique temporary directory under OS temp without external deps.
    let mut dir: PathBuf = std::env::temp_dir();
    let unique = format!(
        "gpt_oss_tok_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    dir.push(unique);
    fs::create_dir_all(&dir).expect("mkdir temp test dir");

    // Case A: scalar eos + pad
    write_file(
        &dir,
        "generation_config.json",
        r#"{ "eos_token_id": 2, "pad_token_id": 0 }"#,
    );
    let ids = load_stop_token_ids(&dir).expect("load ids");
    assert_eq!(ids, vec![0, 2]);

    // Case B: array eos + duplicate pad
    write_file(
        &dir,
        "generation_config.json",
        r#"{ "eos_token_id": [2, 3, 2], "pad_token_id": 0 }"#,
    );
    let ids = load_stop_token_ids(&dir).expect("load ids");
    // Cleanup
    let _ = fs::remove_file(dir.join("generation_config.json"));
    let _ = fs::remove_dir_all(&dir);
    // Should be sorted & deduped
    assert_eq!(ids, vec![0, 2, 3]);
}
