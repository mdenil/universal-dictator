"""Extract a .nemo checkpoint into plain files the Swift converter can read.

Writes, next to the .nemo file, a directory `nemo_extracted/` containing:
  - weights.safetensors   the state_dict with NeMo's own tensor names, fp32
  - model_config.yaml     the model config
  - tokenizer.model etc.  the SentencePiece tokenizer files
  - manifest.json         tensor names/shapes and the sha256 of the source .nemo

Usage: tools/.venv/bin/python tools/export_weights.py models/<model-dir>/<name>.nemo
"""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import torch
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(nemo_path: str) -> None:
    src = Path(nemo_path).resolve()
    out = src.parent / "nemo_extracted"
    out.mkdir(exist_ok=True)
    state = None
    with tarfile.open(src, "r:") as tar:
        for m in tar.getmembers():
            name = Path(m.name).name
            if not m.isfile():
                continue
            data = tar.extractfile(m).read()
            if name == "model_weights.ckpt":
                state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            elif name == "model_config.yaml" or "tokenizer" in name or name.endswith("vocab.txt"):
                # Tokenizer files carry a hash prefix inside the archive; keep a stable name too.
                (out / name).write_bytes(data)
                for stable in ("tokenizer.model", "tokenizer.vocab", "vocab.txt"):
                    if name.endswith(stable):
                        (out / stable).write_bytes(data)
    if state is None:
        raise SystemExit("model_weights.ckpt not found in archive")
    tensors = {k: v.detach().to(torch.float32).contiguous() for k, v in state.items() if torch.is_tensor(v) and v.is_floating_point()}
    ints = {k: v for k, v in state.items() if torch.is_tensor(v) and not v.is_floating_point()}
    save_file(tensors, str(out / "weights.safetensors"))
    manifest = {
        "source": src.name,
        "source_sha256": sha256(src),
        "tensors": {k: list(v.shape) for k, v in sorted(tensors.items())},
        "non_float_tensors_skipped": {k: [list(v.shape), str(v.dtype)] for k, v in sorted(ints.items())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(tensors)} tensors to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
