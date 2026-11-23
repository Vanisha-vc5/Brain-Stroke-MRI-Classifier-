"""Project background, method and limitations."""
import streamlit as st

from src.config import DISCLAIMER, GITHUB_URL

st.set_page_config(page_title="About · Brain Stroke MRI", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About this project")
st.warning(DISCLAIMER, icon="⚠️")

st.markdown(f"""
This app classifies a single brain MRI slice as **Normal**, **Ischemic stroke** or
**Haemorrhagic stroke**, and shows *why* with a Grad-CAM heatmap. It also lists the most
similar training images and lets you download a PDF report.

### How it works
1. The image is converted to RGB, resized to 224 × 224 and normalised with ImageNet statistics.
2. An ImageNet-pretrained **ResNet-18** extracts a 512-dimensional feature vector.
3. A custom head (Dropout 0.4 → Linear 512→256 → ReLU → Dropout 0.2 → Linear 256→3)
   turns it into class probabilities.
4. **Grad-CAM** on the last convolutional block (`layer4[-1]`) highlights the regions that
   pushed the model towards its prediction.
5. The same 512-d vector is compared, using cosine similarity, against 962 training images
   to find the 5 closest look-alikes.

### Training
- **Data:** 1,553 slices from 413 patients, drawn from a Kaggle MRI dataset, ISLES-2022 and IXI.
- **Patient-level split:** 962 train / 307 validation / 284 test images. No patient appears in
  more than one split; the pipeline checks this and stops if it is violated.
- **Two stages:** 10 epochs training only the head (lr 1e-3), then 14 epochs fine-tuning the
  whole network (lr 1e-4) with Adam, `ReduceLROnPlateau` and early stopping.
- **Class imbalance:** a `WeightedRandomSampler` on the training loader only, with plain
  cross-entropy loss (not both, which would double-correct).

### Limitations
- **Haemorrhagic cases come from only 5 patients.** The model caught 50% of haemorrhagic test
  scans; the misses were predicted as Normal.
- **Possible source bias.** Most ischemic images come from ISLES-2022, so the model may partly
  learn the dataset's look rather than the pathology.
- **No clinical validation,** no external test set, and only three categories: mixed,
  atypical or follow-up presentations are out of scope.
- **Any image gets an answer.** The model has no "not an MRI" class, so non-MRI or
  unusual images will still produce a confident-looking prediction.

### Author
**Vanisha Choudhary** · B.Tech Electronics and Communication Engineering,
Netaji Subhas University of Technology (NSUT), Delhi

[Source code on GitHub]({GITHUB_URL})
""")
