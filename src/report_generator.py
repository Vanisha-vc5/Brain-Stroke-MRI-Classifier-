"""PDF report for a single prediction (fpdf2, core fonts only)."""
from datetime import datetime, timezone

from fpdf import FPDF

from src.config import DISCLAIMER, GITHUB_URL


def _latin1(text: str) -> str:
    """Core PDF fonts are latin-1 only; replace anything else."""
    replacements = {"—": "-", "–": "-", "’": "'", "‘": "'",
                    "“": '"', "”": '"', "→": "->", "·": "-"}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf_report(original, overlay, filename, result, class_names,
                     assessment, similar_cases) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Brain Stroke MRI Classifier - Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 5, _latin1(f"Generated {stamp}  |  Image: {filename}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_fill_color(255, 243, 205)
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(0, 5, _latin1("Disclaimer: " + DISCLAIMER), fill=True,
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Prediction", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, _latin1(f"Predicted class: {result['pred_class']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Confidence: {result['confidence']:.1%}  (level: {assessment['level']})",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 7, "Class", border=1)
    pdf.cell(40, 7, "Probability", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for name, p in zip(class_names, result["probs"]):
        pdf.cell(60, 7, name, border=1)
        pdf.cell(40, 7, f"{float(p):.2%}", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin1(assessment["summary"]), new_x="LMARGIN", new_y="NEXT")
    for note in assessment["notes"]:
        pdf.multi_cell(0, 5, _latin1("- " + note), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Image and Grad-CAM", new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y()
    size = 80
    pdf.image(original.convert("RGB"), x=15, y=y, w=size, h=size)
    pdf.image(overlay.convert("RGB"), x=15 + size + 10, y=y, w=size, h=size)
    pdf.set_y(y + size + 2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin1(
        "Left: uploaded image. Right: Grad-CAM overlay for the predicted class (warmer = more "
        "influence on the prediction). A heatmap shows where the model looked; it does not prove "
        "a lesion is present or that the prediction is correct."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    if similar_cases:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Most similar training cases", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 10)
        for header, width in (("#", 10), ("Label", 40), ("Sequence", 30), ("Source", 40), ("Similarity", 30)):
            pdf.cell(width, 7, header, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 10)
        for i, case in enumerate(similar_cases, start=1):
            pdf.cell(10, 7, str(i), border=1)
            pdf.cell(40, 7, _latin1(case["label"]), border=1)
            pdf.cell(30, 7, _latin1(case["sequence"]), border=1)
            pdf.cell(40, 7, _latin1(case["source"]), border=1)
            pdf.cell(30, 7, f"{case['similarity']:.3f}", border=1)
            pdf.ln()
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 4, "Similar cases are nearest neighbours in the model's feature space, "
                             "not confirmed medical matches.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Model", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin1(
        "ImageNet-pretrained ResNet-18 with a custom 3-class head, trained in two stages on 1,553 "
        "MRI slices from 413 patients (Kaggle, ISLES-2022, IXI) with a patient-level split. "
        "Held-out test set (284 images, 62 patients): accuracy 93.66%, macro AUC-ROC 98.55%, "
        "macro F1 87.20%, haemorrhagic recall 50%."), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, _latin1(GITHUB_URL), new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())
