"""Reference streaming runs through the official NeMo streaming code paths (CPU fp32 by default).

Nemotron (cache-aware): calls `perform_streaming` imported from NeMo v2.7.3
  examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py,
  which drives CacheAwareStreamingAudioBuffer + model.conformer_stream_step. The model is set up exactly as
  that script's main() does (att_context_size, RNNTDecodingConfig(fused_batch_size=-1)). Each utterance is
  streamed alone (batch 1). For the first --chunk-utts utterances it also saves, per chunk:
    chunk_XXX_in.npy        [128, n]   mel frames fed to the step (pre-encode cache + new frames)
    chunk_XXX_in_len.npy    int64      the length passed with them
    chunk_XXX_enc.npy       [t, 1024]  encoder output after streaming_post_process (what the decoder sees)
  and the caches returned after chunk 0 and chunk 1:
    cache_after_chunk_K_last_channel.npy [24, 70, 1024], _last_time.npy [24, 1024, 8], _len.npy int64
  plus stream_enc_out.npy (all chunk outputs concatenated) and tokens.npy / transcript.txt.

Parakeet (buffered): runs the official buffered script (c42e444 version, via run_official.py) with
  left_context_secs=5.6 chunk_secs=0.16 right_context_secs=0.40 att_context_size_as_chunk=true,
  batch_size=1, compute_dtype=float32, then splits its output manifest into per-utterance transcripts.
  The dev20 golden was produced with --device mps: the buffered pipeline re-encodes a 6.16 s window every
  160 ms and ran at about 1/20 of real time on this machine's CPU (versus about 1/2 on MPS).

Layout: data/golden/<model>/stream_<cfg>/<utt_id>/...  and data/golden/<model>/stream_<cfg>/transcripts.jsonl

Usage:
  scripts/heavy.sh reference/nemo/.venv/bin/python reference/nemo/stream_ref.py --model nemotron --config 70_1
  scripts/heavy.sh reference/nemo/.venv/bin/python reference/nemo/stream_ref.py --model parakeet --config 70_2_5
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODEL_PATHS, ROOT, THIRD_PARTY, load_model, read_manifest, set_threads, utt_id  # noqa: E402

CA_SCRIPT = (
    THIRD_PARTY / "nemo-v2.7.3-examples/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
)


def import_cache_aware_script():
    spec = importlib.util.spec_from_file_location("nemo_ca_streaming_infer_v273", CA_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_nemotron(cfg_name: str, recs, out_dir: Path, chunk_utts: int):
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    ca = import_cache_aware_script()
    acs = [int(x) for x in cfg_name.split("_")]
    model = load_model("nemotron")
    # As in the script's main(): set lookahead, then the decoding strategy, then eval.
    model.encoder.set_default_att_context_size(att_context_size=acs)
    dec_cfg = RNNTDecodingConfig(fused_batch_size=-1)
    model.change_decoding_strategy(dec_cfg)
    model = model.to(device="cpu", dtype=torch.float32)
    model.eval()
    print("streaming_cfg:", model.encoder.streaming_cfg, flush=True)

    # Capture hooks (instance-level wrappers; the wrapped functions are called unchanged).
    cap = {"enc": [], "steps": []}
    orig_post = model.encoder.streaming_post_process
    orig_step = model.conformer_stream_step

    def post(rets, keep_all_outputs=True):
        r = orig_post(rets, keep_all_outputs=keep_all_outputs)
        cap["enc"].append((r[0].detach().clone(), r[1].detach().clone()))
        return r

    def step(**kw):
        r = orig_step(**kw)
        cap["hyp"] = r[5]
        cap["steps"].append(
            {
                "in": kw["processed_signal"].detach().clone(),
                "in_len": kw["processed_signal_length"].detach().clone(),
                "ch": r[2].detach().clone() if r[2] is not None else None,
                "tm": r[3].detach().clone() if r[3] is not None else None,
                "chlen": r[4].detach().clone() if r[4] is not None else None,
            }
        )
        return r

    model.encoder.streaming_post_process = post
    model.conformer_stream_step = step

    results = []
    t0 = time.time()
    for i, rec in enumerate(recs):
        uid = utt_id(rec)
        cap["enc"].clear()
        cap["steps"].clear()
        buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=False)
        buf.append_audio_file(str(ROOT / rec["audio"]), stream_id=-1)
        tran, _ = ca.perform_streaming(
            asr_model=model,
            streaming_buffer=buf,
            compute_dtype=torch.float32,
            compare_vs_offline=False,
            debug_mode=False,
            pad_and_drop_preencoded=False,
        )
        text = tran[0]
        d = out_dir / uid
        d.mkdir(parents=True, exist_ok=True)
        encs = [e[0, :, : int(n[0])].transpose(0, 1).numpy().astype(np.float32) for e, n in cap["enc"]]
        np.save(d / "stream_enc_out.npy", np.concatenate(encs, axis=0))
        np.save(d / "chunk_enc_lens.npy", np.array([e.shape[0] for e in encs], dtype=np.int64))
        (d / "transcript.txt").write_text(text + "\n")
        np.save(d / "tokens.npy", np.array([int(x) for x in cap["hyp"][0].y_sequence], dtype=np.int64))
        if i < chunk_utts:
            for k, (s, e) in enumerate(zip(cap["steps"], encs)):
                np.save(d / f"chunk_{k:03d}_in.npy", s["in"][0].numpy().astype(np.float32))
                np.save(d / f"chunk_{k:03d}_in_len.npy", np.array(int(s["in_len"][0]), dtype=np.int64))
                np.save(d / f"chunk_{k:03d}_enc.npy", e)
            for k in (0, 1):
                s = cap["steps"][k]
                np.save(d / f"cache_after_chunk_{k}_last_channel.npy", s["ch"][:, 0].numpy().astype(np.float32))
                np.save(d / f"cache_after_chunk_{k}_last_time.npy", s["tm"][:, 0].numpy().astype(np.float32))
                np.save(d / f"cache_after_chunk_{k}_len.npy", np.array(int(s["chlen"][0]), dtype=np.int64))
        results.append({"utt_id": uid, "audio": rec["audio"], "text": rec["text"], "pred_text": text,
                        "n_chunks": len(encs)})
        print(f"[stream_{cfg_name}] {uid}: {text}", flush=True)
    print(f"[stream_{cfg_name}] {len(recs)} utts in {time.time() - t0:.1f}s", flush=True)
    return results


def run_parakeet(cfg_name: str, recs, out_dir: Path, device: str):
    assert cfg_name == "70_2_5", cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)
    man = out_dir / "manifest.nemo.jsonl"
    with open(man, "w") as f:
        for rec in recs:
            f.write(json.dumps({"audio_filepath": str(ROOT / rec["audio"]), "text": rec["text"],
                                "duration": rec["duration"]}) + "\n")
    pred = out_dir / "official_output.jsonl"
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "run_official.py"), "buffered_rnnt",
        f"model_path={MODEL_PATHS['parakeet']}", f"dataset_manifest={man}", f"output_filename={pred}",
        "left_context_secs=5.6", "chunk_secs=0.16", "right_context_secs=0.40", "att_context_size_as_chunk=true",
        "batch_size=1", "compute_dtype=float32", "cuda=-1",
        f"allow_mps={'true' if device == 'mps' else 'false'}",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    results = []
    by_path = {r["audio_filepath"]: r for r in read_manifest(pred)}
    for rec in recs:
        uid = utt_id(rec)
        r = by_path[str(ROOT / rec["audio"])]
        d = out_dir / uid
        d.mkdir(parents=True, exist_ok=True)
        (d / "transcript.txt").write_text(r["pred_text"] + "\n")
        results.append({"utt_id": uid, "audio": rec["audio"], "text": rec["text"], "pred_text": r["pred_text"]})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["nemotron", "parakeet"])
    ap.add_argument("--config", required=True, help="nemotron: 70_1 | 70_6; parakeet: 70_2_5")
    ap.add_argument("--utts", default=str(DATA / "dev" / "dev20.jsonl"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk-utts", type=int, default=2, help="utterances for which per-chunk tensors are saved")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="parakeet only")
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    set_threads(args.threads)
    out_root = Path(args.out_root) if args.out_root else DATA / "golden" / args.model
    out_dir = out_root / f"stream_{args.config}"
    recs = read_manifest(args.utts)
    if args.model == "nemotron":
        res = run_nemotron(args.config, recs, out_dir, args.chunk_utts)
    else:
        res = run_parakeet(args.config, recs, out_dir, args.device)
    with open(out_dir / "transcripts.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
