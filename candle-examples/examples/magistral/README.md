# magistral

An example using Magistral (Mistral 3 multimodal) with Candle. It loads the Tekken tokenizer from `tekken.json`, the model config and weights from the Hugging Face Hub, and the model’s `SYSTEM_PROMPT.txt`. The example prepares a minimal chat sequence using the Mistral instruct template with a single `[IMG]` placeholder, then runs generation while inserting image embeddings via the mistral3 projector.

Usage examples (CUDA, with default model id):

- cargo run --profile=release-with-debug --features tekken,cuda --example magistral -- --image ./test-image.jpg --prompt "Summarize this image."
- cargo run --release --features tekken --example magistral -- --cpu --image ./test-image.jpg --prompt "Summarize this image."

Notes
- The example fetches the following from the model snapshot: `config.json`, `model.safetensors.index.json` (+ shards), `tekken.json`, `SYSTEM_PROMPT.txt`.
- Image preprocessing preserves aspect ratio: the longest side is 1540, both dimensions are rounded down to multiples of 14, and Pixtral mean/std normalization is applied.
- Tokenization: special token ids are parsed from `tekken.json` (no hardcoding), normal text is encoded via `tekken-rs`.
- Generation: on the first step, the model is called with `pixel_values` and `image_sizes`; subsequent steps use text-only.
