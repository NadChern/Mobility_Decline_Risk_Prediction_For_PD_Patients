"""Artifact contract shared by the producer (export.py) and the consumer
(data_loader.py).

Keeping the filenames and key column names in one lightweight module (no heavy
imports such as shap) means the explanation runtime and the modelling notebook
can never drift on what the artifacts are called.
"""

# Filenames written into explanation_artifacts/ and read back by the pipeline.
ARTIFACT_FILENAMES = {
    "shap_stage1": "shap_values_stage1.npy",
    "x_test": "X_test.csv",
    "y_pred_proba": "y_pred_proba.npy",
    "patient_ids": "patient_ids_test.csv",
    "routing_mask": "routing_mask.npy",
    "shap_stage2": "shap_values_stage2.npy",
    "y_pred_s2": "y_pred_s2.npy",
    "y_pred_final": "y_pred_final.npy",
}

# Patient identifier column expected in patient_ids_test.csv.
PATIENT_ID_COLUMN = "PATNO"

# Optional provenance file written alongside the artifacts. NOT required by the
# consumer (data_loader does not load it) — it exists purely to record which
# model/data/run produced the current artifacts.
MANIFEST_FILENAME = "manifest.json"
