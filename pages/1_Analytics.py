"""Model performance on the held-out test set and training history."""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_state import get_model
from src.config import CLASS_COLORS, DISCLAIMER, MANIFEST_PATH, METRICS_PATH, SOURCE_LABELS

st.set_page_config(page_title="Analytics · Brain Stroke MRI", page_icon="📊", layout="wide")

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
_, meta = get_model()

st.title("📊 Model analytics")
test = metrics["test_set"]
st.caption(
    f"Held-out test set: {test['images']} images from {test['patients']} patients, "
    "never seen in training (patient-level split)."
)

overall = metrics["overall"]
cols = st.columns(5)
for col, (label, key) in zip(cols, [
    ("Accuracy", "accuracy"), ("Macro AUC-ROC", "macro_auc_roc"), ("Macro F1", "macro_f1"),
    ("Macro precision", "macro_precision"), ("Macro recall", "macro_recall"),
]):
    col.metric(label, f"{overall[key]:.2%}")

# ------------------------------------------------------------ per class
st.subheader("Per-class performance")
per_class = metrics["per_class"]
rows = [
    {"Class": c, "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1"],
     "Test images": v["support"]}
    for c, v in per_class.items()
]
df = pd.DataFrame(rows)
left, right = st.columns([1, 1.2])
with left:
    shown = df.copy()
    for column in ("Precision", "Recall", "F1"):
        shown[column] = shown[column].map(lambda v: f"{v:.1%}")
    st.dataframe(shown, hide_index=True, width="stretch")
    st.markdown(
        "**Haemorrhagic recall is 50%.** Half of the haemorrhagic test scans were "
        "predicted as Normal. Overall accuracy hides this, so it is shown explicitly here."
    )
with right:
    long = df.melt(id_vars="Class", value_vars=["Precision", "Recall", "F1"],
                   var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Class", y="Score", color="Metric", barmode="group",
                 text_auto=".0%", range_y=[0, 1.08])
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------- confusion matrix
st.subheader("Confusion matrix (test set)")
cm = metrics["confusion_matrix"]
left, right = st.columns([1.1, 1])
with left:
    fig = px.imshow(
        cm["matrix"], x=cm["labels"], y=cm["labels"], text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Images"),
    )
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
with right:
    st.markdown(
        "- All 169 Normal and all 79 Ischemic test images were classified correctly.\n"
        "- 18 of 36 Haemorrhagic images were predicted as **Normal**: the main failure mode.\n"
        "- No other class was ever predicted as Haemorrhagic, which is why its precision is 100%."
    )
    st.caption(cm["note"])

# ------------------------------------------------------ training curves
history = meta.get("history") or {}
if history:
    st.subheader("Training history")
    st.caption("Epochs 1–10: head only (lr 1e-3). Epochs 11–24: full fine-tuning (lr 1e-4).")
    groups = {
        "Loss": [k for k in history if "loss" in k.lower()],
        "Accuracy / AUC": [k for k in history if "acc" in k.lower() or "auc" in k.lower()],
    }
    cols = st.columns(len([g for g in groups.values() if g]) or 1)
    for col, (title, keys) in zip(cols, [(t, k) for t, k in groups.items() if k]):
        fig = go.Figure()
        for key in keys:
            fig.add_trace(go.Scatter(y=history[key], x=list(range(1, len(history[key]) + 1)),
                                     mode="lines+markers", name=key))
        n = max(len(history[k]) for k in keys)
        if n > 10:
            fig.add_vline(x=10.5, line_dash="dash", line_color="gray")
        fig.update_layout(title=title, height=330, margin=dict(l=0, r=0, t=40, b=0),
                          xaxis_title="Epoch")
        col.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------- dataset
st.subheader("Dataset")
manifest = pd.read_csv(MANIFEST_PATH)
left, right = st.columns(2)
with left:
    counts = manifest.groupby(["mri_class", "split"]).size().reset_index(name="Images")
    fig = px.bar(counts, x="mri_class", y="Images", color="split", text_auto=True,
                 category_orders={"mri_class": list(CLASS_COLORS), "split": ["train", "val", "test"]},
                 labels={"mri_class": "Class", "split": "Split"})
    fig.update_layout(title="Images per class and split", height=360,
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")
with right:
    src_counts = manifest.assign(Source=manifest["source"].map(SOURCE_LABELS).fillna(manifest["source"]))
    src_counts = src_counts.groupby(["Source", "mri_class"]).size().reset_index(name="Images")
    fig = px.bar(src_counts, x="Source", y="Images", color="mri_class", text_auto=True,
                 color_discrete_map=CLASS_COLORS, labels={"mri_class": "Class"})
    fig.update_layout(title="Images per source", height=360, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")
st.caption(
    f"{len(manifest):,} images from {manifest['patient_id'].nunique()} patients. "
    "Each patient appears in exactly one split, so test scores reflect unseen patients."
)

# ---------------------------------------------------------- leakage study
st.subheader("Why the patient-level split matters")
leak = metrics["leakage_study"]
study = pd.DataFrame([
    {"Setup": "Image-level split (leaky)", "Accuracy": leak["image_level_split"]["accuracy"],
     "AUC": leak["image_level_split"]["auc"]},
    {"Setup": "Patient-level split, original data", "Accuracy": leak["patient_level_split_original_data"]["accuracy"],
     "AUC": leak["patient_level_split_original_data"]["auc"]},
    {"Setup": "Patient-level split, final data", "Accuracy": leak["patient_level_split_final_data"]["accuracy"],
     "AUC": leak["patient_level_split_final_data"]["auc"]},
])
fig = px.bar(study.melt(id_vars="Setup", var_name="Metric", value_name="Score"),
             x="Setup", y="Score", color="Metric", barmode="group", text_auto=".1%",
             range_y=[0, 1.08])
fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_tickformat=".0%")
st.plotly_chart(fig, width="stretch")
st.markdown(
    "Randomly splitting slices let images of the same patient land in both training and test, "
    "which inflated accuracy to 96.8%. Grouping by patient dropped it to 77.7% on the original "
    "data; expanding the dataset brought it back to 93.66% on genuinely unseen patients."
)
st.caption(DISCLAIMER)
