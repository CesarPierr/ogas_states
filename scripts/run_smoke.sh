#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m poolbased_surrogate.run configs/smoke.yaml
python -m poolbased_surrogate.run configs/smoke_weighted.yaml
