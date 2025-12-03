#!/usr/bin/env bash
set -euo pipefail
# This script is an example usage of `convert_to_bag.py` to convert the nuScenes mini-v1.0 dataset to MCAP.

if [ ! -d "data" ]; then
    echo "data dir does not exist: please create and extract nuScenes data into it."
    exit 1
fi

docker build -t mcap_converter . -f nuscenes2bag/Dockerfile
mkdir -p output
docker run -t --rm \
    --user $(id -u):$(id -g) \
    -v $(pwd)/data:/data -v $(pwd)/output:/output \
    mcap_converter python3 nuscenes2bag/convert_to_bag.py --data-dir /data --output-dir /output "$@"