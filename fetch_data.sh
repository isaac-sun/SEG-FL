#!/usr/bin/env bash
# Prepare VOC 2012 aug and the ImageNet backbone weights, once.
#
# Looks for each input in this order and only downloads what is missing:
#   1. already extracted under dataset/raw/
#   2. an archive you uploaded into dataset/raw/
#        VOCtrainval_11-May-2012.tar, SegmentationClassAug.zip (or SBD benchmark.tgz)
#   3. AutoDL public data (/root/autodl-pub): VOC tar or extracted VOC2012, SegmentationClassAug
#   4. download (on AutoDL `source /etc/network_turbo` is tried first); a source slower
#      than MIN_SPEED (default 300 kB/s) for a minute is dropped for the next one
# Then builds dataset/voc_seg/*.npy with prepare_data.py (about 10 minutes).
# Backbone: pretrained/mobilenet_v3_large-8738ca79.pth (ships in the zip;
# downloaded from download.pytorch.org only if missing).
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAW="${RAW_DIR:-dataset/raw}"
OUT="${DATA_DIR:-dataset/voc_seg}"
VOC_URL="${VOC_URL:-https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar}"
VOC_MIRROR="${VOC_MIRROR:-http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar}"
AUG_URL="${AUG_URL:-https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=1}"
SBD_URL="${SBD_URL:-http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz}"
WEIGHTS="pretrained/mobilenet_v3_large-8738ca79.pth"
WEIGHTS_URL="https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth"
mkdir -p "$RAW" pretrained
export PYTHONPATH="$PROJECT_DIR/pip_pkgs${PYTHONPATH:+:$PYTHONPATH}"

if [ -f "$OUT/meta.json" ] && [ -f "$OUT/train_images.npy" ]; then
    echo "data ready: $OUT"; NEED_DATA=0
else
    NEED_DATA=1
fi
if [ -f /etc/network_turbo ]; then source /etc/network_turbo >/dev/null 2>&1 || true; fi

fetch() {   # fetch <dest> <url> [<url> ...]
    local dest="$1"; shift
    for url in "$@"; do
        echo "download $url"
        # Give up on a source slower than MIN_SPEED bytes/s for a minute and try the next.
        if curl -fL --retry 3 --connect-timeout 20 \
                --speed-limit "${MIN_SPEED:-300000}" --speed-time 60 \
                -o "$dest.part" "$url"; then
            mv "$dest.part" "$dest"; return 0
        fi
    done
    rm -f "$dest.part"; return 1
}

if [ ! -s "$WEIGHTS" ]; then
    fetch "$WEIGHTS" "$WEIGHTS_URL" || {
        echo "could not get $WEIGHTS; upload it from your computer into pretrained/" >&2; exit 1; }
fi
"$PYTHON_BIN" -c "import torch; sd=torch.load('$WEIGHTS', map_location='cpu', weights_only=True); assert any(k.startswith('features.') for k in sd)" \
    || { echo "$WEIGHTS is not a MobileNetV3-Large state dict" >&2; exit 1; }
echo "backbone weights ok: $WEIGHTS"
[ "$NEED_DATA" = 0 ] && exit 0

# VOC 2012 trainval
VOC_ROOT="$RAW/VOCdevkit/VOC2012"
if [ ! -d "$VOC_ROOT/JPEGImages" ]; then
    TAR="$RAW/VOCtrainval_11-May-2012.tar"
    if [ ! -f "$TAR" ]; then
        PUB="$(find /root/autodl-pub /root/autodl-tmp -maxdepth 5 -name 'VOCtrainval_11-May-2012.tar' 2>/dev/null | head -1 || true)"
        PUB_DIR="$(find /root/autodl-pub -maxdepth 6 -type d -path '*VOC2012/JPEGImages' 2>/dev/null | head -1 || true)"
        if [ -n "$PUB" ]; then echo "using local copy $PUB"; TAR="$PUB"
        elif [ -n "$PUB_DIR" ]; then
            echo "using AutoDL public folder $(dirname "$PUB_DIR")"
            mkdir -p "$RAW/VOCdevkit"; ln -sfn "$(dirname "$PUB_DIR")" "$VOC_ROOT"; TAR=""
        else fetch "$TAR" "$VOC_URL" "$VOC_MIRROR" || {
            echo "could not download VOC 2012; upload VOCtrainval_11-May-2012.tar into $RAW/" >&2; exit 1; }
        fi
    fi
    if [ -n "$TAR" ]; then echo "extract $TAR"; tar -xf "$TAR" -C "$RAW"; fi
fi

# Augmented labels (SegmentationClassAug), else the SBD release
AUG_ARGS=()
if [ ! -d "$RAW/SegmentationClassAug" ] && [ ! -f "$RAW/SegmentationClassAug.zip" ]; then
    PUB_AUG="$(find /root/autodl-pub -maxdepth 6 \( -name 'SegmentationClassAug.zip' -o -type d -name SegmentationClassAug \) 2>/dev/null | head -1 || true)"
    if [ -d "$PUB_AUG" ]; then ln -sfn "$PUB_AUG" "$RAW/SegmentationClassAug"
    elif [ -n "$PUB_AUG" ]; then cp "$PUB_AUG" "$RAW/"; fi
fi
if [ ! -d "$RAW/SegmentationClassAug" ] && [ -f "$RAW/SegmentationClassAug.zip" ]; then
    "$PYTHON_BIN" -m zipfile -e "$RAW/SegmentationClassAug.zip" "$RAW"
fi
if [ ! -d "$RAW/SegmentationClassAug" ] && [ ! -d "$RAW/benchmark_RELEASE" ] && [ ! -f "$RAW/benchmark.tgz" ]; then
    if fetch "$RAW/SegmentationClassAug.zip" "$AUG_URL"; then
        "$PYTHON_BIN" -m zipfile -e "$RAW/SegmentationClassAug.zip" "$RAW"
    else
        fetch "$RAW/benchmark.tgz" "$SBD_URL" || true
    fi
fi
if [ -d "$RAW/SegmentationClassAug" ]; then
    AUG_ARGS=(--aug-labels "$RAW/SegmentationClassAug")
else
    if [ ! -d "$RAW/benchmark_RELEASE" ] && [ -f "$RAW/benchmark.tgz" ]; then
        tar -xzf "$RAW/benchmark.tgz" -C "$RAW"
    fi
    if [ -d "$RAW/benchmark_RELEASE/dataset" ]; then
        AUG_ARGS=(--sbd-root "$RAW/benchmark_RELEASE/dataset")
    else
        echo "no augmented labels found: upload SegmentationClassAug.zip into $RAW/ " \
             "(continuing without them would leave only 1,464 training images)" >&2
        exit 1
    fi
fi

"$PYTHON_BIN" prepare_data.py --voc-root "$VOC_ROOT" "${AUG_ARGS[@]}" --out "$OUT" \
    --workers "${WORKERS:-$(nproc 2>/dev/null || echo 8)}"
echo "data ready: $OUT"
