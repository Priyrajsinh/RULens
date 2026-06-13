# RULens

> Opening a lens onto a remaining-useful-life forecaster: mechanistic
> interpretability and calibrated uncertainty for time-series foundation models,
> applied to industrial predictive maintenance.

RULens studies what time-series foundation models (Chronos, TimesFM) actually
compute when they forecast remaining useful life on industrial sensor data
(NASA C-MAPSS, Bosch CNC). It stacks three methods: **sparse autoencoders** to
surface interpretable internal features, **intervention-based faithfulness**
(activation patching / ablation) to test which of those features *causally*
drive the forecast, and **conformal prediction** to attach calibrated intervals
to the point forecast.

## Status

Scaffolding. The shared infrastructure (`src/common`), configuration, API
schemas, CI gates, and project guards are in place; the research modules are
stubbed and implemented day by day.

## Quickstart

```bash
python -m venv venv
# Windows:  venv\Scripts\activate     Linux/macOS:  source venv/bin/activate
pip install -r requirements-dev.txt
pip install numpy pyyaml pydantic     # lightweight runtime for the gates

# Run the quality gates (format, lint, types, security, tests + coverage):
make gates PY=venv/Scripts/python.exe   # Windows
make gates                              # Linux/macOS (venv activated)
```

The full experiment stack (PyTorch, Transformers, Chronos, sae_lens, MAPIE,
EconML, MLflow, DVC) is pinned in `requirements.txt` and installed on GPU
machines for the heavy training and interpretability runs.

## Layout

| Path | Purpose |
|------|---------|
| `src/common/` | Config, structured logging, exceptions, deterministic seeding |
| `src/data/` | C-MAPSS / Bosch loaders, RUL labeling, windowing |
| `src/models/` | TSFM wrappers, baselines, activation-capture hooks |
| `src/sae/` | Sparse-autoencoder training and architectures |
| `src/interp/` | Feature gallery, monosemanticity scoring |
| `src/faithfulness/` | Activation patching and ablation (primary causal evidence) |
| `src/conformal/` | Split-CP, EnbPI, SPCI, CPTC |
| `src/causal/` | Double-ML attribution (secondary) |
| `src/api/`, `src/ui/` | FastAPI service and Streamlit dashboard |

## Reproducibility

Every reported number traces to an MLflow run; datasets, activation caches, and
checkpoints are versioned with DVC. All hyperparameters live in
`config/config.yaml` — there are no magic numbers in `src/`.

## Citation

A BibTeX entry will be added on preprint submission.

## License

MIT.
