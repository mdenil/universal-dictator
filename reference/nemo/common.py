"""Shared helpers for the NeMo reference scripts (Nemotron and Parakeet Unified).

Test infrastructure only. Everything runs on CPU in fp32 unless a script says otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ASRModel.restore_from unpacks the 2.4 GB .nemo into a temporary directory; keep that inside the repo.
_TMP = ROOT / "scratch" / "tmp"
_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_TMP)
import tempfile  # noqa: E402

tempfile.tempdir = str(_TMP)

DATA = ROOT / "data"
MODELS = ROOT / "models"
THIRD_PARTY = ROOT / "third_party"

MODEL_PATHS = {
    "nemotron": MODELS / "nemotron-speech-streaming-en-0.6b" / "nemotron-speech-streaming-en-0.6b.nemo",
    "parakeet": MODELS / "parakeet-unified-en-0.6b" / "parakeet-unified-en-0.6b.nemo",
}

# NeMo main commit that added `chunked_limited_with_rc` / `att_chunk_context_size` / `conv_context_style`
# (PR #15522). NeMo 2.7.3 (the pinned wheel) does not contain it, so Parakeet Unified cannot be
# instantiated by 2.7.3 alone. See docs/specs/parakeet.md, "Runtime: NeMo 2.7.3 cannot load this model".
UNIFIED_COMMIT = "c42e4445127650492f96a7b47a39a4f449fdcf50"
UNIFIED_ENCODER_SRC = THIRD_PARTY / "NeMo" / "nemo" / "collections" / "asr" / "modules" / "conformer_encoder.py"

_SHIM_INSTALLED = False


def install_unified_shim():
    """Patch the 2.7.3 ConformerEncoder with the upstream c42e444 methods needed by Parakeet Unified.

    The patched methods are taken verbatim (as code objects, not re-typed) from the upstream file at
    commit c42e444, checked out in third_party/NeMo. Between v2.7.3 and c42e444 the only behavioural
    differences in conformer_encoder.py are: the two new constructor arguments, the
    `chunked_limited_with_rc` mask branch in `_create_masks`, the `[[-1,-1,-1]]` default in
    `_calc_context_sizes` and a `chunked_limited_with_rc` branch in `setup_streaming_params`.
    All other asr modules used here (features, subsampling, conformer_modules, multi_head_attention,
    rnnt, label looping) are behaviourally identical between the two versions.
    """
    global _SHIM_INSTALLED
    if _SHIM_INSTALLED:
        return
    import nemo.collections.asr.modules.conformer_encoder as ce273

    if not UNIFIED_ENCODER_SRC.exists():
        raise FileNotFoundError(
            f"{UNIFIED_ENCODER_SRC} missing: check out NeMo {UNIFIED_COMMIT} (sparse) in third_party/NeMo"
        )
    spec = importlib.util.spec_from_file_location("_nemo_c42_conformer_encoder", UNIFIED_ENCODER_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    up = mod.ConformerEncoder
    cls = ce273.ConformerEncoder

    orig_init = cls.__init__

    def __init__(self, *args, att_chunk_context_size=None, conv_context_style="regular", **kwargs):
        # c42e444 conformer_encoder.py:358-372 (verbatim semantics): validate and store.
        assert conv_context_style in ["regular", "dcc"], f"Invalid conv_context_style: {conv_context_style}!"
        att_context_style = kwargs.get("att_context_style", "regular")
        if att_chunk_context_size is not None:
            assert att_context_style == "chunked_limited_with_rc"
            assert len(att_chunk_context_size) == 3
        orig_init(self, *args, **kwargs)
        self.conv_context_style = conv_context_style
        self.conv_kernel_size = kwargs.get("conv_kernel_size", 31)
        self.att_chunk_context_size = att_chunk_context_size

    cls.__init__ = __init__
    cls._create_masks = up._create_masks
    cls._calc_context_sizes = up._calc_context_sizes
    cls.setup_streaming_params = up.setup_streaming_params
    # up.setup_streaming_params refers to module globals (CacheAwareStreamingConfig, MultiHeadAttention,
    # CausalConv1D, logging): those resolve in the upstream module, which imported them from the same
    # 2.7.3 packages, so isinstance checks still match.
    _SHIM_INSTALLED = True


def set_threads(n: int):
    import torch

    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)


def load_model(name: str, device: str = "cpu"):
    """Load a model from its local .nemo file, eval mode, fp32, dither 0."""
    import torch
    from nemo.collections.asr.models import ASRModel

    if name == "parakeet":
        install_unified_shim()
    path = MODEL_PATHS[name]
    model = ASRModel.restore_from(str(path), map_location=torch.device(device))
    model.eval()
    model = model.to(torch.float32)
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    return model


def read_manifest(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def utt_id(rec: dict) -> str:
    return Path(rec["audio"]).stem


def load_audio(path: str):
    """Read 16 kHz mono float32 audio with soundfile (the same reader NeMo's get_samples uses)."""
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    assert sr == 16000, sr
    assert x.ndim == 1
    return np.ascontiguousarray(x, dtype=np.float32)
