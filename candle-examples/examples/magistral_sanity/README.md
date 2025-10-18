magistral_sanity

Runs a best‑effort sanity check between the Python Transformers pipeline and the Candle Magistral example.

- Compares tokenization length and verifies a single `[IMG]` placeholder is present.
- Compares pixel preprocessing grid `(H/patch, W/patch)`.
- Generates 16 tokens in both pipelines and prints both texts plus a simple word‑set Jaccard similarity.

Examples
- cargo run --release --features tekken --example magistral_sanity -- --cpu --image ./test-image.jpg
- cargo run --features tekken,cuda --example magistral_sanity -- --image ./test-image.jpg --new-tokens 16

Requires `uv` and the local Python example `infer_with_magistral.py` present in the repo.

