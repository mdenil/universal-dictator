#!/bin/bash
# M3/M2 sweep on 200-utterance subsets (fp16): Parakeet exact vs frozen-prefix horizons, Nemotron 160/560.
cd "$(dirname "$0")/.."
B=.build/release/dictate-cli
for set in test-clean test-other; do
  for spec in parakeet parakeet-h0 parakeet-h4 parakeet-h16 nemotron-160 nemotron-560; do
    name=${spec/parakeet-h/parakeet_h}; name=${name/parakeet/parakeet}; name=${name/nemotron-/nemotron_}
    [ "$spec" == "parakeet" ] && name=parakeet_exact
    $B eval --model $spec --precision f16 --manifest data/librispeech/$set.jsonl --out artifacts/eval/${name}_f16_${set}_200.jsonl --limit 200
  done
done
reference/nemo/.venv/bin/python eval/score.py artifacts/eval/*_200.jsonl 2>/dev/null
