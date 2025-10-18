#!/usr/bin/env python3
from typing import Any
import os
import torch
from pathlib import Path

from transformers import Mistral3ForConditionalGeneration, AutoTokenizer
import argparse
import json


# Simple, minimal constants
MODEL_ID = "mistralai/magistral-small-2509"  # match Rust crate default
IMAGE_PATH = Path("./test-image.jpg").resolve()
TEMP = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 512  # default; can be overridden with CLI

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
    parser = argparse.ArgumentParser(description="Magistral sanity helper")
    parser.add_argument("--image", type=str, default=str(IMAGE_PATH), help="Path to input image")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMP)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--meta-json", action="store_true", help="Print metadata as JSON to stdout")
    parser.add_argument("--prompt", type=str, default="Summarize this slide image comprehensively.")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available")
    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")

    snapshot = _find_local_snapshot(args.model_id)

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), tokenizer_type="mistral", use_fast=False, local_files_only=True
    )
    model = Mistral3ForConditionalGeneration.from_pretrained(
        str(snapshot), dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()

    system_prompt = load_system_prompt_from_snapshot(snapshot, "SYSTEM_PROMPT.txt")

    # Use a local file URL for the image
    image_url = f"file://{image_path}"
    messages = [
        system_prompt,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    tokenized = tokenizer.apply_chat_template(messages, return_dict=True)

    input_ids = torch.tensor(tokenized.input_ids, device="cuda").unsqueeze(0)
    attention_mask = torch.tensor(tokenized.attention_mask, device="cuda").unsqueeze(0)
    pixel_values = torch.tensor(tokenized.pixel_values[0], dtype=torch.bfloat16, device="cuda").unsqueeze(0)
    image_sizes = torch.tensor(pixel_values.shape[-2:], device="cuda").unsqueeze(0)

    # Compute patch grid dims
    patch_size = int(getattr(getattr(model.config, "vision_config", object()), "patch_size", 14))
    h, w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
    grid_h, grid_w = h // patch_size, w // patch_size

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

    # Strip the prompt from the decoded output
    trimmed = output[len(tokenized.input_ids) : (-1 if output[-1] == tokenizer.eos_token_id else len(output))]
    text = tokenizer.decode(trimmed)

    if args.meta_json:
        meta = {
            "input_ids_len": len(tokenized.input_ids),
            "image_h": h,
            "image_w": w,
            "patch_size": patch_size,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "new_tokens": len(trimmed),
            "text": text,
        }
        print(json.dumps(meta))
    else:
        print(text)


if __name__ == "__main__":
    main()
