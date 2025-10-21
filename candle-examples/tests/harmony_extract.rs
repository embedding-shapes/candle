use gpt_oss_tokenizer::extract_final_assistant_text_from_decoded;

// T7.2 Harmony output extraction
// Stream tokens but emit only the assistant's "final" channel message content.
// This test constructs a decoded Harmony string that includes an analysis channel
// followed by a final channel, and asserts that only the final content is extracted
// byte-for-byte.
#[test]
fn extract_final_only_from_decoded_text() {
    // Synthetic decoded text consistent with Harmony chat template markers.
    // The final text intentionally contains punctuation and newlines to verify
    // byte-for-byte extraction without trimming or control tokens.
    let decoded = concat!(
        "<|start|>user<|message|>Explain MXFP4.<|end|>",
        "<|start|>assistant<|channel|>analysis<|message|>",
        "Thinking about FP4 quantization...\nIt uses E2M1 with per-block E8M0 scales.",
        "<|end|>",
        "<|start|>assistant<|channel|>final<|message|>",
        "MXFP4 is a 4-bit floating-point format (E2M1) ",
        "with 8-bit exponent-only (E8M0) per-block scales; ",
        "together they preserve range with minimal bits.",
        "<|end|>"
    );
    let expected = "MXFP4 is a 4-bit floating-point format (E2M1) \
with 8-bit exponent-only (E8M0) per-block scales; together they preserve range with minimal bits.";
    let got = extract_final_assistant_text_from_decoded(decoded).expect("must extract final text");
    assert_eq!(got, expected);
}

// Also ensure that if another assistant header appears after the final (e.g., a tool call),
// we still stop before any control token and never include it.
#[test]
fn extract_stops_before_control_tokens() {
    let decoded = concat!(
        "<|start|>assistant<|channel|>final<|message|>",
        "Answer here.",
        "<|return|>", // control token must not be included
        "<|start|>assistant<|channel|>analysis<|message|>postlude<|end|>"
    );
    let got = extract_final_assistant_text_from_decoded(decoded).expect("must extract");
    assert_eq!(got, "Answer here.");
}
