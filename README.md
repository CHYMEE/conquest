# CONQUEST: revision rerun

This branch holds the complete rerun of the CONQUEST experiments under the leakage-free protocol.

| Folder / file | Contents |
|---|---|
| `revision/*.py` | pipeline: data and context features, QPSO/PSO/variance selectors, centralized and federated training, statistics, tables and figures |
| `revision/grids/` | experiment grids (blocks A-H) |
| `revision/outputs/` | result file of every run (JSON/CSV) and the statistics computed from them |
| `revision/cache/` | data-split report and context-feature self-test |
| `revision/response/` | sources of the response letter |
| `qubit_ablation_study.py` | QPSO selector, LSTM(+VAE) model, and metrics classes imported by the pipeline |

See `revision/README_revision.md` for the protocol and the exact commands.
The dataset (ACI IoT Network Traffic Dataset 2023, IEEE DataPort, DOI 10.21227/qacj-3x32)
is not included; save its NetFlow CSV as `0.ACI-IoT-2023.csv` in the repository root.
Model weights and data caches are not included; the commands regenerate them.
