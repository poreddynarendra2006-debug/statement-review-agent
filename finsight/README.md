# `finsight/`

**Owner:** Anomaly

The anomaly detection package: Isolation Forest and robust z-scores over 27 scale-neutral features, with a plain-English explanation for every finding.

| Folder | What it contains |
|:--|:--|
| `analysis/` | Feature engineering, dataset profiling, the detector, scoring and explanations |
| `core/` | Configuration and data models, including `AnomalyFinding` |
| `data/` | Dataset loading |

The pipeline reaches it through `analysis/anomaly_detection.py`. The model is trained on each upload, so no trained model file is stored.
