"""Run an official NeMo example script unmodified, in this venv (NeMo 2.7.3).

Usage:
  reference/nemo/.venv/bin/python reference/nemo/run_official.py {cache_aware|buffered_rnnt} [hydra overrides...]

cache_aware    -> NeMo v2.7.3 examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
                  (third_party/nemo-v2.7.3-examples, fetched from the v2.7.3 tag; the wheel does not ship examples)
buffered_rnnt  -> NeMo c42e444 examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py
                  (third_party/NeMo sparse checkout). This is the v2.7.3 script plus the 12-line
                  `att_context_size_as_chunk` block that Parakeet Unified needs. Before running it we install the
                  c42e444 ConformerEncoder shim (common.install_unified_shim), because the 2.7.3 wheel cannot
                  instantiate Parakeet Unified's encoder.
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import THIRD_PARTY, install_unified_shim  # noqa: E402

SCRIPTS = {
    "cache_aware": THIRD_PARTY
    / "nemo-v2.7.3-examples/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py",
    "buffered_rnnt": THIRD_PARTY
    / "NeMo/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py",
}


def main():
    which = sys.argv[1]
    script = SCRIPTS[which]
    if which == "buffered_rnnt":
        install_unified_shim()
    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
