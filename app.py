"""Brain Stroke MRI Classifier: upload a slice, get a prediction with Grad-CAM."""
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.analysis import confidence_assessment
from src.app_state import get_gallery, get_model
from src.config import (CLASS_COLORS, DISCLAIMER, EXAMPLES_DIR, GITHUB_URL,
                        SOURCE_LABELS, TOP_K_SIMILAR)
from src.explainability import grad_cam, overlay_heatmap
from src.predict import colour_score, predict
from src.report_generator import build_pdf_report
from src.retrieval import most_similar

st.set_page_config(page_title="Brain Stroke MRI Classifier", page_icon="🧠", layout="wide")

model, meta = get_model()
class_names = meta["class_names"]

with st.sidebar:
    st.markdown("### About this model")
    st.markdown(
        "ImageNet-pretrained **ResNet-18**, fine-tuned to classify brain MRI slices as "
        "**Normal**, **Ischemic** or **Haemorrhagic**."
    )
    st.markdown(
        "- Test accuracy: **93.66%**\n"
        "- Macro AUC-ROC: **98.55%**\n"
        "- Haemorrhagic recall: **50%**\n\n"
        "Tested on 284 held-out images from 62 patients."
    )
    st.markdown(f"[Source code on GitHub]({GITHUB_URL})")
    st.caption(DISCLAIMER)

st.title("🧠 Brain Stroke MRI Classifier")
st.caption("ResNet-18 · Normal / Ischemic / Haemorrhagic · Grad-CAM explanations · similar-case search")
st.warning(DISCLAIMER, icon="⚠️")

# ---------------------------------------------------------------- input
source = st.radio("Choose an image", ["Upload my own", "Try an example"], horizontal=True)
image, image_name, true_label = None, None, None

if source == "Upload my own":
    upload = st.file_uploader("Upload a brain MRI slice (PNG or JPG)", type=["png", "jpg", "jpeg"])
    if upload is not None:
        try:
            image = Image.open(upload).convert("RGB")
            image_name = upload.name
        except Exception:
            st.error("That file couldn't be opened as an image. Try a PNG or JPG.")
else:
    examples = sorted(EXAMPLES_DIR.glob("*.png"))
    options = {}
    for path in examples:
        label, _, number = path.stem.rpartition("_")
        options[f"{label.title()} example {number}"] = (path, label.title())
    if options:
        choice = st.selectbox(
            "Example scans (held-out test images the model never trained on)", list(options)
        )
        path, true_label = options[choice]
        image = Image.open(path).convert("RGB")
        image_name = path.name

if image is None:
    st.info("Upload an MRI slice or pick an example to get a prediction.")
    st.stop()

if max(image.size) > 2048:
    image.thumbnail((2048, 2048))

if colour_score(image) > 25:
    st.warning(
        "This image looks like a colour photo rather than a greyscale MRI slice. "
        "The model will still give an answer, but it is only meaningful for brain MRI."
    )

# ------------------------------------------------------------ inference
result = predict(model, image, class_names)
cam = grad_cam(model, result["tensor"], result["pred_idx"])
overlay = overlay_heatmap(image, cam)
assessment = confidence_assessment(result["probs"], class_names)
pred = result["pred_class"]
color = CLASS_COLORS.get(pred, "#444")

col_img, col_cam, col_result = st.columns([1, 1, 1.1])
with col_img:
    st.markdown("**Uploaded image**")
    st.image(image, width="stretch")
with col_cam:
    st.markdown("**Grad-CAM heatmap**")
    st.image(overlay, width="stretch")
    st.caption("Warmer areas influenced the prediction more.")
with col_result:
    st.markdown("**Prediction**")
    st.markdown(
        f"<div style='border-left:6px solid {color};padding:0.4rem 0.9rem;"
        f"border-radius:4px;background:rgba(127,127,127,0.08)'>"
        f"<div style='font-size:1.9rem;font-weight:700;color:{color}'>{pred}</div>"
        f"<div>{result['confidence']:.1%} probability · {assessment['level']} confidence</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if true_label:
        verdict = "matches" if true_label == pred else "does not match"
        st.caption(f"True label of this example: **{true_label}** (prediction {verdict}).")

    fig = go.Figure(go.Bar(
        x=[float(p) for p in result["probs"]],
        y=class_names,
        orientation="h",
        marker_color=[CLASS_COLORS.get(c, "#888") for c in class_names],
        text=[f"{float(p):.1%}" for p in result["probs"]],
        textposition="auto",
    ))
    fig.update_layout(
        height=190, margin=dict(l=0, r=10, t=10, b=0),
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# --------------------------------------------------------- assessment
st.subheader("What this result means")
st.markdown(assessment["summary"])
for note in assessment["notes"]:
    st.markdown(f"- {note}")
st.caption(
    "Any real scan should be read by a radiologist. This tool is a research demo, "
    "not a diagnostic device."
)

# -------------------------------------------------------- similar cases
st.subheader("Similar training cases")
with st.spinner("Building the similar-case index (first run only, about a minute)..."):
    gallery = get_gallery(model)
neighbours = most_similar(result["embedding"], gallery["embeddings"], k=TOP_K_SIMILAR)
similar_cases = []
if neighbours:
    records = gallery["records"]
    cols = st.columns(len(neighbours))
    for col, (idx, sim) in zip(cols, neighbours):
        rec = records.iloc[idx]
        case = {
            "label": rec["mri_class"],
            "sequence": str(rec["sequence"]),
            "source": SOURCE_LABELS.get(rec["source"], str(rec["source"])),
            "similarity": sim,
        }
        similar_cases.append(case)
        with col:
            st.image(Image.open(rec["abs_path"]), width="stretch")
            st.markdown(
                f"<span style='color:{CLASS_COLORS.get(case['label'], '#444')};font-weight:600'>"
                f"{case['label']}</span><br><small>{case['sequence']} · {case['source']}<br>"
                f"similarity {sim:.3f}</small>",
                unsafe_allow_html=True,
            )
    agree = sum(c["label"] == pred for c in similar_cases)
    st.caption(
        f"{agree} of {len(similar_cases)} nearest training images are labelled {pred}. "
        "Similarity is cosine similarity between 512-d ResNet-18 embeddings: these are "
        "look-alikes in the model's feature space, not confirmed medical matches."
    )
else:
    st.info("Similar-case search is unavailable: no training images were found.")

# ---------------------------------------------------------------- report
st.subheader("Download report")
pdf_bytes = build_pdf_report(image, overlay, image_name, result, class_names,
                             assessment, similar_cases)
st.download_button(
    "📄 Download PDF report",
    data=pdf_bytes,
    file_name=f"stroke_mri_report_{pred.lower()}.pdf",
    mime="application/pdf",
)
