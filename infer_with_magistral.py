#!/usr/bin/env python3
from typing import Any
import os
import torch
from pathlib import Path

from transformers import Mistral3ForConditionalGeneration, AutoTokenizer


# Simple, minimal constants
MODEL_ID = "mistralai/magistral-small-2509"  # match Rust crate default
IMAGE_PATH = Path("./test-image.jpg").resolve()
TEMP = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 512  # keep short for a quick sanity-check

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _hub_root() -> Path:
    # Prefer HF_HOME / HF_HUB_CACHE if set; otherwise default path
    p = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    return Path(p).expanduser() if p else Path.home() / ".cache" / "huggingface" / "hub"


def _title_case_repo(repo: str) -> str:
    parts = repo.split("-")
    return "-".join(s[:1].upper() + s[1:] if s else s for s in parts)


def _find_local_snapshot(repo_id: str) -> Path:
    org, repo = repo_id.split("/", 1)
    hub = _hub_root()
    candidates = [
        hub / f"models--{org}--{repo}" / "snapshots",
        hub / f"models--{org}--{_title_case_repo(repo)}" / "snapshots",
    ]
    snaps: list[Path] = []
    for root in candidates:
        if root.is_dir():
            snaps.extend([p for p in root.iterdir() if p.is_dir()])
    if not snaps:
        raise FileNotFoundError(f"no local snapshots found for {repo_id} under {hub}")
    # Prefer snapshots that look complete (safetensors shards present)
    def is_complete(s: Path) -> bool:
        idx = s / "model.safetensors.index.json"
        if not idx.is_file():
            return False
        # At least one shard file present
        for p in s.iterdir():
            name = p.name
            if name.startswith("model-") and name.endswith(".safetensors") and p.is_file():
                return True
        return False

    snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for s in snaps:
        if is_complete(s):
            return s
    # Fallback to most recent even if incomplete
    return snaps[0]


def load_system_prompt_from_snapshot(snapshot_dir: Path, filename: str) -> dict[str, Any]:
    file_path = snapshot_dir / filename
    with open(file_path, "r") as file:
        system_prompt = file.read()

    i0 = system_prompt.find("[THINK]")
    i1 = system_prompt.find("[/THINK]")
    return {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt[:i0]},
            {"type": "thinking", "thinking": system_prompt[i0 + len("[THINK]") : i1], "closed": True},
            {"type": "text", "text": system_prompt[i1 + len("[/THINK]") :]},
        ],
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available")
    if not IMAGE_PATH.is_file():
        raise SystemExit(f"image not found: {IMAGE_PATH}")

    snapshot = _find_local_snapshot(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), tokenizer_type="mistral", use_fast=False, local_files_only=True
    )
    model = Mistral3ForConditionalGeneration.from_pretrained(
        str(snapshot), dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()

    system_prompt = load_system_prompt_from_snapshot(snapshot, "SYSTEM_PROMPT.txt")

    # Use a local file URL for the image
    image_url = f"file://{IMAGE_PATH}"
    messages = [
        system_prompt,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this slide image comprehensively."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    tokenized = tokenizer.apply_chat_template(messages, return_dict=True)

    input_ids = torch.tensor(tokenized.input_ids, device="cuda").unsqueeze(0)
    attention_mask = torch.tensor(tokenized.attention_mask, device="cuda").unsqueeze(0)
    pixel_values = torch.tensor(tokenized.pixel_values[0], dtype=torch.bfloat16, device="cuda").unsqueeze(0)
    image_sizes = torch.tensor(pixel_values.shape[-2:], device="cuda").unsqueeze(0)

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=TEMP,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

    # Strip the prompt from the decoded output
    trimmed = output[len(tokenized.input_ids) : (-1 if output[-1] == tokenizer.eos_token_id else len(output))]
    print(tokenizer.decode(trimmed))


if __name__ == "__main__":
    main()
