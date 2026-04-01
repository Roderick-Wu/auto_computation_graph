#!/bin/bash
set -euo pipefail

for experiment in "velocity" "current" "radius" "side_length" "wavelength" "cross_section" "displacement" "market_cap"
do
    bash post_process.sh "$experiment" "Qwen2.5-32B"
done

echo "All post-processing jobs submitted."