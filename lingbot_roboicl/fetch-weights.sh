#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$PWD
export HF_HOME="$root/cache/hf" HF_XET_CACHE="$root/cache/xet" HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TMPDIR="$root/tmp"
mkdir -p checkpoints
.venv/bin/hf download robbyant/lingbot-va-base \
    --revision 68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c --local-dir checkpoints/base \
    --include 'transformer/*' 'vae/*' 'text_encoder/*' 'tokenizer/*' --max-workers 4
