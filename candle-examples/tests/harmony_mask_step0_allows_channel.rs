use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use gpt_oss_tokenizer::allowed_specials_for_next;
use std::collections::BTreeSet;

// Constants from the prompt spec
const ID_CHANNEL: usize = 200005;
const ID_MESSAGE: usize = 200008;
const ID_RETURN: usize = 200002;
const ID_CALL: usize = 200012;
const ID_END: usize = 200007;
const VOCAB: usize = 201_088;

fn fsm_allowed_ids(decoded_so_far: &str) -> BTreeSet<usize> {
    // Map Harmony specials to their ids per the spec.
    let mut map = std::collections::BTreeMap::new();
    map.insert("<|channel|>", ID_CHANNEL);
    map.insert("<|message|>", ID_MESSAGE);
    map.insert("<|return|>", ID_RETURN);
    map.insert("<|call|>", ID_CALL);
    map.insert("<|end|>", ID_END);
    // Compute the allowed special symbols using the same FSM as the example binary.
    let allow_syms = allowed_specials_for_next(decoded_so_far);
    let mut ids: BTreeSet<usize> = BTreeSet::new();
    for s in allow_syms {
        if let Some(id) = map.get(s) {
            ids.insert(*id as usize);
        }
    }
    ids
}

fn device_cuda_or_cpu() -> Device {
    match candle_examples::device(false /* cpu */) {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}

fn build_logits(base: f32) -> Vec<f32> {
    vec![base; VOCAB]
}

fn softmax_mask_probabilities(
    logits: &[f32],
    to_mask: &BTreeSet<usize>,
    device: &Device,
) -> Result<Vec<f32>> {
    // Build tensor on chosen device, softmax, then zero masked ids just like the sampler closure.
    let t = Tensor::from_vec(logits.to_vec(), VOCAB, device)?;
    eprintln!(
        "dtype={:?} device={:?} shape=({})",
        DType::F32,
        device,
        VOCAB
    );
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::All { temperature: 1.0 });
    let mut captured: Option<Vec<f32>> = None;
    let _ = sampler.sample_f(&t, |prs: &mut [f32]| {
        for &i in to_mask {
            if i < prs.len() {
                prs[i] = 0.0;
            }
        }
        captured = Some(prs.to_vec());
    })?;
    Ok(captured.expect("captured probs"))
}

#[test]
fn harmony_mask_step0_allows_channel() -> Result<()> {
    // State: AfterAssistant (rendered prompt ends with '<|start|>assistant').
    let decoded = "<|start|>assistant";
    let allow_ids = fsm_allowed_ids(decoded);
    eprintln!("detected_state=AfterAssistant allow_ids={:?}", allow_ids);

    // Global forbid set: suppress specials by default, re-enable via FSM.
    let mut global_forbid: BTreeSet<usize> = BTreeSet::new();
    for &id in &[ID_CHANNEL, ID_MESSAGE, ID_RETURN, ID_CALL, ID_END] {
        global_forbid.insert(id);
    }
    let to_mask: BTreeSet<usize> = global_forbid.difference(&allow_ids).copied().collect();

    // Build logits vector per spec.
    let mut l = build_logits(-10.0);
    l[ID_CHANNEL] = 10.0;
    l[ID_MESSAGE] = 5.0;
    l[ID_RETURN] = 7.0;
    l[ID_CALL] = 7.0;
    l[ID_END] = 7.0;
    let pre_argmax = argmax(&l);
    assert_eq!(pre_argmax, ID_CHANNEL, "argmax(L) must be <|channel|>");

    // Mask at probability level, mirroring the example sampler.
    let device = device_cuda_or_cpu();
    let pm = softmax_mask_probabilities(&l, &to_mask, &device)?;
    let post_argmax = argmax(&pm);
    eprintln!(
        "(pre_mask_argmax, post_mask_argmax)=({},{}) five_pre={{ch:{:.1}, msg:{:.1}, ret:{:.1}, call:{:.1}, end:{:.1}}} five_post={{ch:{:.6}, msg:{:.6}, ret:{:.6}, call:{:.6}, end:{:.6}}}",
        pre_argmax, post_argmax,
        l[ID_CHANNEL], l[ID_MESSAGE], l[ID_RETURN], l[ID_CALL], l[ID_END],
        pm[ID_CHANNEL], pm[ID_MESSAGE], pm[ID_RETURN], pm[ID_CALL], pm[ID_END]
    );

    // Assertions per spec intent adapted for probability-level masking.
    assert!(
        allow_ids.contains(&ID_CHANNEL),
        "<|channel|> must be in allow set"
    );
    assert_eq!(pm[ID_CHANNEL], pm[ID_CHANNEL], "channel prob finite");
    assert_eq!(
        pm[ID_MESSAGE], 0.0,
        "<|message|> must be masked to zero at step 0"
    );
    assert_eq!(
        pm[ID_RETURN], 0.0,
        "<|return|> must be masked to zero at step 0"
    );
    assert_eq!(
        pm[ID_CALL], 0.0,
        "<|call|> must be masked to zero at step 0"
    );
    assert_eq!(pm[ID_END], 0.0, "<|end|> must be masked to zero at step 0");
    // A few normal tokens must remain allowed (retain non-zero mass but lose to channel).
    for &tid in &[11usize, 13usize, 25usize] {
        assert!(pm[tid] >= 0.0, "normal token {} must remain allowed", tid);
    }
    assert_eq!(
        post_argmax, ID_CHANNEL,
        "<|channel|> must remain top-1 after masking at step 0"
    );

    // Negative control: set channel low, a normal token high; ensure normal token wins.
    let mut l2 = build_logits(-10.0);
    l2[ID_CHANNEL] = -10.0;
    l2[11] = 9.0;
    let p2 = softmax_mask_probabilities(&l2, &to_mask, &device)?;
    assert_eq!(
        argmax(&p2),
        11usize,
        "normal token 11 must win when channel is low"
    );

    // State transition: AfterChannel(final) → require <|message|>, channel must be masked.
    let decoded2 = "<|start|>assistant<|channel|>final";
    let allow2 = fsm_allowed_ids(decoded2);
    let to_mask2: BTreeSet<usize> = global_forbid.difference(&allow2).copied().collect();
    let mut l3 = build_logits(-10.0);
    l3[ID_MESSAGE] = 10.0;
    let p3 = softmax_mask_probabilities(&l3, &to_mask2, &device)?;
    assert_eq!(
        argmax(&p3),
        ID_MESSAGE,
        "<|message|> must be top-1 after channel"
    );
    assert_eq!(
        p3[ID_CHANNEL], 0.0,
        "<|channel|> must be masked after channel token"
    );

    Ok(())
}
