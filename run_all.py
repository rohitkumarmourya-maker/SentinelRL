#!/usr/bin/env python3
"""Show the current champion entry points; no accidental historical-v2 pipeline."""
print('Trained demonstration: python demo.py --steps 48')
print('Evidence/replay: python scripts/verify_champion.py --replay')
print('Fresh training reproduction: python train.py --execute-model-run --name reproduction_champion')
print('Quick bundled-policy reevaluation: python evaluate.py --execute-model-run --output runs/quick-check')
print('Full fixed benchmark reproduction: python evaluate.py --execute-model-run --full-frozen-test --workers 8 --output runs/full-check')
