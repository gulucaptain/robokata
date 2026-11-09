#!/usr/bin/env bash
set -euo pipefail

source_dir=$(cd "$(dirname "$0")" && pwd)
if [[ $# != 1 || "$1" != /* ]]; then
    echo "Usage: bash bootstrap.sh /absolute/path/to/new/deployment" >&2
    exit 2
fi
root=$1
uv=$(command -v uv || true)
uv=${uv:-"$HOME/.local/bin/uv"}
test -x "$uv"
mkdir "$root"
mkdir "$root"/{src,roboicl,tmp,logs,cache}
cp "$source_dir"/*.py "$root/roboicl/"
for launcher in job_roboicl.sh run_roboicl.sh publish_roboicl.sh fetch-weights.sh; do
    install -m 700 "$source_dir/$launcher" "$root/$launcher"
done
cp "$source_dir"/{SOURCE_REV,environment.lock} "$root/"
cp "$source_dir/README.md" "$root/ROBOICL.md"
cd "$root"
export TMPDIR="$root/tmp" UV_CACHE_DIR="$root/cache/uv" UV_LINK_MODE=hardlink UV_HTTP_TIMEOUT=120
rev=$(<SOURCE_REV)
curl -fsSL --retry 2 "https://codeload.github.com/robbyant/lingbot-va/tar.gz/$rev" |
    tar -xz --strip-components=1 -C src
python3 roboicl/patch_upstream.py src
"$uv" venv --python 3.10 --no-python-downloads .venv
# Install the recorded CUDA wheel set and LeRobot without resolving additional dependencies.
"$uv" pip install --python .venv/bin/python --no-deps \
    --index 'https://download.pytorch.org/whl/cu126' --index-strategy unsafe-best-match \
    -r environment.lock
.venv/bin/python -m py_compile roboicl/*.py src/wan_va/{train,wan_va_server,modules/model}.py
