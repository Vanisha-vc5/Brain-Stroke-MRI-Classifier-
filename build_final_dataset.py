"""
Builds dataset/final_dataset/ -- a harmonized, patient-grouped, leakage-safe
dataset combining:
  1. The original Stroke_classification/{Normal,Ischemic,Haemorrhagic} 2D slices
  2. ISLES-2022 (250 real ischemic-stroke patients, NIfTI volumes + lesion masks)
  3. IXI-2022 (150 real healthy-control patients, Normal only)
  4. VALDO-2021 Task 2 (real cerebral-microbleed patients, T2* + CMB masks) --
     added to Haemorrhagic as extra train/val diversity only. Microbleeds are
     a different (smaller, chronic) lesion type than acute hemorrhagic
     stroke, so these patients are deliberately excluded from the test
     split -- see TEST_EXCLUDED_SOURCES / process_valdo().

Why this exists (see conversation history / README_dataset.md for full detail):
  - The original dataset had only ~16 unique patients total (5 Ischemic, 5
    Haemorrhagic, ~6 Normal) with many near-duplicate slices per patient, and
    the notebook's train_test_split worked at image level -> patient leakage
    across train/val/test, producing inflated (not trustworthy) metrics.
  - The original images are NOT skull-stripped and carry burned-in scanner
    annotations (e.g. "[AH]", "GE MEDICAL") near the image border/head -- a
    second, subtler leakage vector, since annotation style can correlate
    with source/class.
  - ISLES-2022 only contains ischemic cases (no Normal, no Haemorrhagic, no
    GRE/SWI), so it can only help the Ischemic class -- but it does so
    massively: 250 real patients vs. 5.

This script:
  - Harmonizes every image (both sources) through the same pipeline:
    grayscale morphological opening (kills thin burned-in text without
    erasing thick anatomical structures) -> light gaussian smoothing ->
    largest-connected-component brain crop -> percentile intensity
    normalization -> resize.
  - Extracts a small, capped number of lesion-containing DWI slices per
    ISLES patient (not all ~40+ per volume) to avoid one source dominating
    via near-duplicate slices.
  - Assigns every image a patient_id and greedily allocates whole patients
    (never individual images) to train/val/test per class, so no patient's
    slices ever appear in more than one split.
  - Writes dataset/final_dataset/<Class>/<file>.png + manifest.csv with
    columns: path,label,mri_class,patient_id,source,sequence,split
"""
import os
import re
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import nibabel as nib

random.seed(42)
np.random.seed(42)

# ROOT is this script's own directory (the Stroke_prediction project root), not a
# hardcoded machine path -- the original ROOT (C:\stroke_prediction_jupyter_nb) was
# where this pipeline was first run; that machine's raw source folders don't ship
# with this repo, only their processed output (dataset/final_dataset/) does. Any
# source whose raw folder is missing here relies entirely on its cached
# manifest_*.csv fragment (see _run_cached / the migration step in main()) --
# it will error out if asked to reprocess from scratch without that raw folder.
ROOT = os.path.dirname(os.path.abspath(__file__))
OLD_DATASET = os.path.join(ROOT, "dataset", "Stroke_classification")
ISLES_DIR = os.path.join(ROOT, "ISLES-2022")
IXI_DIR = os.path.join(ROOT, "IXI_dataset")
VALDO_DIR = os.path.join(ROOT, "dataset", "VALDO_Task2", "Task2")
OUT_DIR = os.path.join(ROOT, "dataset", "final_dataset")
IMG_SIZE = 256
ISLES_SLICES_PER_PATIENT = 2  # cap to avoid ISLES dominating via near-duplicates
VALDO_SLICES_PER_PATIENT = 2  # same reasoning -- cap per-subject near-duplicates

# 150 of 578 IXI patients, stratified proportionally across the 3 hospital
# sites (319/185/74 in the full set -> ~83/48/19 at n=150). Only Normal
# has no lesion mask to guide slice choice -- these are healthy controls,
# so slices are picked by position instead (mid-brain, avoiding the top/
# bottom 20% of the volume where there's little brain tissue).
IXI_N_PATIENTS = 150
IXI_SITE_TARGETS = {"Guys": 83, "HH": 48, "IOP": 19}
IXI_SLICES_PER_PATIENT = 3

CLASS_TO_LABEL = {"Normal": 0, "Ischemic": 1, "Haemorrhagic": 2}
SEQ_TOKENS = ["T2 Flair", "T2Flair", "DWI", "GRE", "SWI", "T2", "FLAIR", "ADC"]


# ------------------------------------------------------------------ #
# Shared harmonization pipeline
# ------------------------------------------------------------------ #
def harmonize(gray_arr, margin=4, working_size=256):
    """Crop to the largest connected bright region (the head/brain) and
    normalize intensity, on a common working resolution.

    Every pixel kept in the output is the ORIGINAL scan value -- no
    blurring, no morphological smoothing, no inpainting/fill of any region.
    An earlier version of this function ran a grayscale morphological
    opening across the whole image to erase burned-in scanner text, then
    (in an intermediate, since-reverted attempt) inpainted flagged pixels
    with a local median. Both altered real pixel values -- the opening
    posterized healthy tissue into flat blocky patches, and the inpainting
    fabricated plausible-looking texture in place of pixels we don't
    actually have data for. Neither belongs in a dataset that's meant to
    reflect the real scan. The only transformation here is masking: pixels
    outside the head's connected region (background, and any watermark
    that sits outside the head silhouette) are zeroed, exactly like the
    black background already surrounding every scan. A watermark that
    happens to directly overlap the head silhouette itself is NOT
    specially removed -- doing so would require guessing at pixel values
    we don't have, i.e. fabrication. That's an accepted, disclosed
    limitation, not something to paper over.

    The working canvas is fixed because source resolutions vary a lot
    (ISLES DWI is natively 112x112, the original dataset is 256x256 or
    512x512) -- cropping/masking behave more consistently on a common
    resolution than on wildly different native ones."""
    gray_arr = gray_arr.astype(np.float64)
    if gray_arr.shape != (working_size, working_size):
        gray_arr = np.array(
            Image.fromarray(gray_arr).resize((working_size, working_size), Image.BICUBIC),
            dtype=np.float64,
        )

    thresh = max(15.0, np.percentile(gray_arr, 60) * 0.25)
    fg = gray_arr > thresh
    fg = ndimage.binary_opening(fg, structure=np.ones((3, 3)))
    labeled, n = ndimage.label(fg)
    if n == 0:
        cropped = gray_arr
    else:
        sizes = ndimage.sum(fg, labeled, range(1, n + 1))
        biggest = np.argmax(sizes) + 1
        head_mask = ndimage.binary_fill_holes(labeled == biggest)
        cleaned = np.where(head_mask, gray_arr, 0)
        ys, xs = np.where(head_mask)
        if len(ys) == 0:
            cropped = cleaned
        else:
            y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, cleaned.shape[0])
            x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, cleaned.shape[1])
            cropped = cleaned[y0:y1, x0:x1]

    if cropped.size == 0 or cropped.max() <= 0:
        cropped = gray_arr  # degenerate fallback, keep something sane

    nonzero = cropped[cropped > 0]
    if nonzero.size > 100:
        lo, hi = np.percentile(nonzero, [1, 99])
    else:
        lo, hi = cropped.min(), cropped.max()
    hi = max(hi, lo + 1e-6)
    rescaled = np.clip((cropped - lo) / (hi - lo), 0, 1) * 255.0
    rescaled = np.where(cropped <= 0, 0, rescaled)  # keep true background black

    out = Image.fromarray(rescaled.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
    return out.convert("RGB")


def patient_id_from_filename(fname):
    for tok in SEQ_TOKENS:
        idx = fname.find(f" {tok}-")
        if idx == -1:
            idx = fname.find(f" {tok} ")
        if idx != -1:
            return fname[:idx].strip()
    # fallback: strip trailing digits/extension
    return re.sub(r"[\d\-\._].*$", "", fname).strip() or fname


# ------------------------------------------------------------------ #
# Step 1: harmonize the existing 2D dataset
# ------------------------------------------------------------------ #
def process_existing_dataset(manifest_rows):
    for cls, label in CLASS_TO_LABEL.items():
        src_dir = os.path.join(OLD_DATASET, cls)
        out_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(src_dir, "*")))
        print(f"[existing] {cls}: {len(files)} files")
        for fpath in files:
            fname = os.path.basename(fpath)
            patient = patient_id_from_filename(fname)
            gray = np.array(Image.open(fpath).convert("L"))
            harmonized = harmonize(gray)
            out_name = f"{patient.replace(' ', '_')}__{os.path.splitext(fname)[0]}.png"
            out_name = re.sub(r"[^\w\-.]", "_", out_name)
            out_path = os.path.join(out_dir, out_name)
            harmonized.save(out_path)
            manifest_rows.append({
                "path": os.path.relpath(out_path, ROOT).replace("\\", "/"),
                "label": label,
                "mri_class": cls,
                "patient_id": f"orig_{patient}",
                "source": "kaggle_slices",
                "sequence": next((t for t in SEQ_TOKENS if f" {t}-" in fname or f" {t} " in fname), "unknown"),
            })


# ------------------------------------------------------------------ #
# Step 2: extract capped lesion slices from ISLES-2022
# ------------------------------------------------------------------ #
def robust_normalize_volume(vol):
    nz = vol[vol > 0]
    if nz.size < 100:
        return vol
    lo, hi = np.percentile(nz, [1, 99])
    hi = max(hi, lo + 1e-6)
    return np.clip((vol - lo) / (hi - lo), 0, 1) * 255.0


def process_isles(manifest_rows):
    out_dir = os.path.join(OUT_DIR, "Ischemic")
    os.makedirs(out_dir, exist_ok=True)
    subs = sorted(d for d in os.listdir(ISLES_DIR) if d.startswith("sub-strokecase"))
    print(f"[ISLES] {len(subs)} subjects found")
    kept, skipped = 0, 0

    for sub in subs:
        dwi_path = os.path.join(ISLES_DIR, sub, "ses-0001", "dwi", f"{sub}_ses-0001_dwi.nii.gz")
        mask_path = os.path.join(ISLES_DIR, "derivatives", sub, "ses-0001", f"{sub}_ses-0001_msk.nii.gz")
        if not (os.path.exists(dwi_path) and os.path.exists(mask_path)):
            skipped += 1
            continue

        dwi = nib.load(dwi_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        lesion_area = mask.sum(axis=(0, 1))
        lesion_slices = np.where(lesion_area > 0)[0]
        if len(lesion_slices) == 0:
            skipped += 1
            continue

        # pick up to N slices, spread across the lesion's extent (not all
        # adjacent near-duplicates) -- e.g. 25th and 75th percentile index
        if len(lesion_slices) >= 2:
            picks_idx = np.percentile(np.arange(len(lesion_slices)),
                                       np.linspace(20, 80, ISLES_SLICES_PER_PATIENT)).astype(int)
            picks = lesion_slices[np.clip(picks_idx, 0, len(lesion_slices) - 1)]
            picks = sorted(set(picks.tolist()))
        else:
            picks = [int(lesion_slices[0])]

        dwi_norm = robust_normalize_volume(dwi)
        for k, sl in enumerate(picks):
            slice_2d = dwi_norm[:, :, sl]
            harmonized = harmonize(slice_2d)
            out_name = f"ISLES_{sub}_slice{sl:03d}.png"
            out_path = os.path.join(out_dir, out_name)
            harmonized.save(out_path)
            manifest_rows.append({
                "path": os.path.relpath(out_path, ROOT).replace("\\", "/"),
                "label": CLASS_TO_LABEL["Ischemic"],
                "mri_class": "Ischemic",
                "patient_id": f"ISLES_{sub}",
                "source": "isles2022",
                "sequence": "dwi",
            })
            kept += 1

    print(f"[ISLES] kept {kept} slices from {len(subs) - skipped} usable subjects "
          f"({skipped} skipped: missing files or empty mask)")


# ------------------------------------------------------------------ #
# Step 3: extract position-based mid-brain slices from a stratified
# sample of IXI-2022 healthy-control T2 volumes (Normal class)
# ------------------------------------------------------------------ #
def process_ixi(manifest_rows):
    out_dir = os.path.join(OUT_DIR, "Normal")
    os.makedirs(out_dir, exist_ok=True)

    subject_dirs = sorted(d for d in os.listdir(IXI_DIR) if d.endswith(".nii"))
    # site is embedded in the folder name, e.g. "IXI022-Guys-0821-T2.nii"
    by_site = {"Guys": [], "HH": [], "IOP": []}
    for d in subject_dirs:
        m = re.match(r"(IXI\d+)-(Guys|HH|IOP)-", d)
        if m:
            by_site[m.group(2)].append(d)
    print(f"[IXI] available by site: " +
          ", ".join(f"{s}={len(v)}" for s, v in by_site.items()))

    rng = random.Random(42)
    sampled = []
    for site, target_n in IXI_SITE_TARGETS.items():
        pool = sorted(by_site[site])
        rng.shuffle(pool)
        sampled.extend(pool[:target_n])
    print(f"[IXI] sampled {len(sampled)} patients "
          f"(target {IXI_N_PATIENTS}): " +
          ", ".join(f"{s}={sum(1 for d in sampled if f'-{s}-' in d)}" for s in IXI_SITE_TARGETS))

    kept, skipped = 0, 0
    for sub_dir in sampled:
        subject_id = re.match(r"(IXI\d+)-", sub_dir).group(1)
        inner_files = glob.glob(os.path.join(IXI_DIR, sub_dir, "*.nii"))
        if len(inner_files) != 1:
            skipped += 1
            continue
        nii_path = inner_files[0]

        try:
            vol = nib.load(nii_path).get_fdata()
        except Exception:
            skipped += 1
            continue

        n_slices = vol.shape[2]
        lo, hi = int(n_slices * 0.2), int(n_slices * 0.8)  # skip top/bottom 20%
        if hi <= lo:
            skipped += 1
            continue

        positions = np.linspace(lo, hi, IXI_SLICES_PER_PATIENT).astype(int)
        picks = sorted(set(positions.tolist()))

        vol_norm = robust_normalize_volume(vol)
        for sl in picks:
            slice_2d = vol_norm[:, :, sl]
            harmonized = harmonize(slice_2d)
            out_name = f"{subject_id}_slice{sl:03d}.png"
            out_path = os.path.join(out_dir, out_name)
            harmonized.save(out_path)
            manifest_rows.append({
                "path": os.path.relpath(out_path, ROOT).replace("\\", "/"),
                "label": CLASS_TO_LABEL["Normal"],
                "mri_class": "Normal",
                "patient_id": f"IXI_{subject_id}",
                "source": "ixi_normal",
                "sequence": "T2",
            })
            kept += 1

    print(f"[IXI] kept {kept} slices from {len(sampled) - skipped} usable patients "
          f"({skipped} skipped: missing/unreadable volume)")


# ------------------------------------------------------------------ #
# Step 3b: extract capped microbleed slices from VALDO-2021 Task 2
# (cerebral microbleeds) -- adds real MRI patients with real blood-
# breakdown signal on a T2*-family sequence to the Haemorrhagic class.
#
# IMPORTANT CAVEAT (not fabricated away): a cerebral microbleed is a
# tiny, chronic, punctate lesion -- visually very different from the
# large acute hemorrhagic-stroke bleeds elsewhere in the Haemorrhagic
# class. Checked directly against a few subjects here, some lesions are
# under 100 voxels total in a 512x512x35 volume. This is added as extra
# train/val diversity (more real brains, more real blood-sensitive
# signal), NOT as a claim that it looks like the target pathology.
# assign_splits() below keeps every VALDO-sourced patient out of the
# test split for exactly this reason -- test accuracy should still mean
# "catches real hemorrhagic strokes," not "catches microbleeds."
#
# VALDO Task 2 also includes negative (0-microbleed) subjects -- those
# are skipped here (same as an ISLES subject with an empty mask), since
# labeling a bleed-free scan "Haemorrhagic" would be a mislabel, not
# useful negative data.
# ------------------------------------------------------------------ #
def process_valdo(manifest_rows):
    out_dir = os.path.join(OUT_DIR, "Haemorrhagic")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(VALDO_DIR):
        print(f"[VALDO] directory not found, skipping: {VALDO_DIR}")
        return
    subs = sorted(d for d in os.listdir(VALDO_DIR) if d.startswith("sub-"))
    print(f"[VALDO] {len(subs)} subjects found")
    kept, skipped = 0, 0

    for sub in subs:
        t2s_path = os.path.join(VALDO_DIR, sub, f"{sub}_space-T2S_desc-masked_T2S.nii.gz")
        mask_path = os.path.join(VALDO_DIR, sub, f"{sub}_space-T2S_CMB.nii.gz")
        if not (os.path.exists(t2s_path) and os.path.exists(mask_path)):
            skipped += 1
            continue

        t2s = nib.load(t2s_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        lesion_area = mask.sum(axis=(0, 1))
        lesion_slices = np.where(lesion_area > 0)[0]
        if len(lesion_slices) == 0:
            skipped += 1  # negative control -- no microbleeds annotated, not usable here
            continue

        if len(lesion_slices) >= 2:
            picks_idx = np.percentile(np.arange(len(lesion_slices)),
                                       np.linspace(20, 80, VALDO_SLICES_PER_PATIENT)).astype(int)
            picks = lesion_slices[np.clip(picks_idx, 0, len(lesion_slices) - 1)]
            picks = sorted(set(picks.tolist()))
        else:
            picks = [int(lesion_slices[0])]

        t2s_norm = robust_normalize_volume(t2s)
        for sl in picks:
            slice_2d = t2s_norm[:, :, sl]
            harmonized = harmonize(slice_2d)
            out_name = f"VALDO_{sub}_slice{sl:03d}.png"
            out_path = os.path.join(out_dir, out_name)
            harmonized.save(out_path)
            manifest_rows.append({
                "path": os.path.relpath(out_path, ROOT).replace("\\", "/"),
                "label": CLASS_TO_LABEL["Haemorrhagic"],
                "mri_class": "Haemorrhagic",
                "patient_id": f"VALDO_{sub}",
                "source": "valdo_cmb",
                "sequence": "T2S",
            })
            kept += 1

    print(f"[VALDO] kept {kept} slices from {len(subs) - skipped} usable subjects "
          f"({skipped} skipped: missing files or no microbleeds annotated)")


# ------------------------------------------------------------------ #
# Step 4: patient-grouped, class-stratified split (no image-level leakage)
# ------------------------------------------------------------------ #
# Sources whose patients must never land in the test split -- they're real
# extra training/validation diversity but not the pathology the reported
# test metric is meant to measure (see process_valdo()'s docstring-comment).
TEST_EXCLUDED_SOURCES = {"valdo_cmb"}


def assign_splits(df, val_frac=0.15, test_frac=0.15, seed=42):
    """Patient-level split with explicit minimum guarantees, since some
    classes here have only 5-6 total patients -- a plain proportional
    image-count target (previous approach) could and did starve the test
    set entirely for those classes. Rule: with >=2 patients, guarantee >=1
    in test; with >=3 patients, also guarantee >=1 in val; everything else
    goes to train. For classes with many patients (e.g. Ischemic's 252)
    this reduces to an ordinary ~70/15/15 patient-level split.

    Patients from TEST_EXCLUDED_SOURCES are never chosen for the test
    slot -- test_frac is applied to the test-eligible pool, not the full
    patient count, so adding an excluded-source patient can only grow
    train/val, never shrink or dilute test. For classes with no excluded
    patients this is a no-op and reduces exactly to the original formula."""
    rng = random.Random(seed)
    df = df.copy()
    df["split"] = ""
    patient_source = df.groupby("patient_id")["source"].first()

    for cls in df["mri_class"].unique():
        cls_df = df[df["mri_class"] == cls]
        patient_list = list(cls_df["patient_id"].unique())
        rng.shuffle(patient_list)
        n = len(patient_list)

        test_eligible = [p for p in patient_list if patient_source.get(p) not in TEST_EXCLUDED_SOURCES]
        n_eligible = len(test_eligible)

        n_test = max(1, round(n_eligible * test_frac)) if n_eligible >= 2 else 0
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        while n_test + n_val >= n and (n_test + n_val) > 0:
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
        n_test = min(n_test, n_eligible)

        test_ids = test_eligible[:n_test]
        remaining = [p for p in patient_list if p not in test_ids]
        val_ids = remaining[:n_val]
        train_ids = remaining[n_val:]

        if n_test == 0 or n_val == 0:
            print(f"  [!] {cls}: only {n} patient(s) -- "
                  f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
                  f"(too few patients for a full 3-way split; metrics on the "
                  f"thin split(s) will be statistically weak)")

        df.loc[df["patient_id"].isin(train_ids), "split"] = "train"
        df.loc[df["patient_id"].isin(val_ids), "split"] = "val"
        df.loc[df["patient_id"].isin(test_ids), "split"] = "test"
    return df


def _run_cached(cache_name, process_fn):
    """Each source is cached separately so adding a new source doesn't force
    reprocessing images (harmonization) for sources already done."""
    cache_path = os.path.join(OUT_DIR, cache_name)
    if os.path.exists(cache_path):
        print(f"Reusing cached manifest fragment -> {cache_path} "
              f"(delete this file to force reprocessing this source)")
        return pd.read_csv(cache_path)
    rows = []
    process_fn(rows)
    frag = pd.DataFrame(rows)
    frag.to_csv(cache_path, index=False)
    return frag


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # one-time migration: split the old combined cache (if present) into
    # per-source fragments so existing/isles don't get reprocessed
    old_combined = os.path.join(OUT_DIR, "manifest_presplit.csv")
    existing_cache = os.path.join(OUT_DIR, "manifest_existing.csv")
    isles_cache = os.path.join(OUT_DIR, "manifest_isles.csv")
    ixi_cache = os.path.join(OUT_DIR, "manifest_ixi.csv")
    if os.path.exists(old_combined) and not (os.path.exists(existing_cache) and os.path.exists(isles_cache)):
        combined = pd.read_csv(old_combined)
        combined[combined["source"] == "kaggle_slices"].to_csv(existing_cache, index=False)
        combined[combined["source"] == "isles2022"].to_csv(isles_cache, index=False)
        print(f"Migrated {old_combined} -> {existing_cache}, {isles_cache}")

    # second-chance migration: a checked-out copy of this repo may only ship
    # the final, already-split manifest.csv (built on a different machine),
    # without the per-source fragment caches -- reconstruct them from it so
    # existing/isles/ixi aren't reprocessed from raw source folders that
    # don't exist here (they'd error out or silently produce 0 rows).
    final_manifest = os.path.join(OUT_DIR, "manifest.csv")
    if os.path.exists(final_manifest):
        full = None
        for cache_path, source in [(existing_cache, "kaggle_slices"),
                                    (isles_cache, "isles2022"),
                                    (ixi_cache, "ixi_normal")]:
            if not os.path.exists(cache_path):
                if full is None:
                    full = pd.read_csv(final_manifest)
                frag = full[full["source"] == source].drop(columns=["split"], errors="ignore")
                if len(frag):
                    frag.to_csv(cache_path, index=False)
                    print(f"Reconstructed {cache_path} from {final_manifest}")

    frag_existing = _run_cached("manifest_existing.csv", process_existing_dataset)
    frag_isles = _run_cached("manifest_isles.csv", process_isles)
    frag_ixi = _run_cached("manifest_ixi.csv", process_ixi)
    frag_valdo = _run_cached("manifest_valdo.csv", process_valdo)

    df = pd.concat([frag_existing, frag_isles, frag_ixi, frag_valdo], ignore_index=True)
    df = assign_splits(df)

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    df.to_csv(manifest_path, index=False)

    print("\n=== FINAL DATASET SUMMARY ===")
    print(f"Total images: {len(df)}")
    print(f"Total unique patients: {df['patient_id'].nunique()}")
    print("\nBy class:")
    print(df.groupby("mri_class")["patient_id"].nunique().rename("patients"))
    print(df["mri_class"].value_counts().rename("images"))
    print("\nBy class x source (image count):")
    print(df.groupby(["mri_class", "source"]).size())
    print("\nBy class x split (image count):")
    print(df.groupby(["mri_class", "split"]).size())
    print("\nBy class x split (patient count):")
    print(df.groupby(["mri_class", "split"])["patient_id"].nunique())

    # Leakage sanity check: no patient_id should appear in more than one split
    leak = df.groupby("patient_id")["split"].nunique()
    n_leaks = (leak > 1).sum()
    print(f"\nPatients spanning >1 split (should be 0): {n_leaks}")
    assert n_leaks == 0, "LEAKAGE DETECTED"

    print(f"\nManifest written -> {manifest_path}")


if __name__ == "__main__":
    main()
