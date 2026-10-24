# Running

Requirements: Python 3.12+, about 1 GB disk for the CPU PyTorch wheel. No GPU, no network after install.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
aifound demo                 # full run, ~30 s on an 8-core laptop
aifound demo --quick --out /tmp/q   # smoke run, a few seconds
pytest -q                    # unit tests (~10 s)
python scripts/count_tests.py
python scripts/render_docs.py        # refresh README/docs numbers after a full run
python scripts/render_docs.py --check
python scripts/check_brands.py
python scripts/overlap_check.py /path/outside/repo/reference.txt   # 8-word overlap, target 0
```

CI runs lint, tests, the printed test count, a quick demo, the docs drift check, the brand check and
gitleaks. CodeQL runs separately. Nothing deploys anywhere.
