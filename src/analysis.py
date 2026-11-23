"""Plain-language reading of the class probabilities."""
import numpy as np


def confidence_assessment(probs, class_names) -> dict:
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(probs)[::-1]
    top, second = probs[order[0]], probs[order[1]]
    margin = top - second
    pred = class_names[order[0]]
    runner_up = class_names[order[1]]

    if top >= 0.85 and margin >= 0.5:
        level = "High"
        summary = f"The model is confident this scan is {pred}."
    elif top >= 0.6:
        level = "Moderate"
        summary = f"The model leans towards {pred}, with {runner_up} as the next most likely class."
    else:
        level = "Low"
        summary = f"The model is uncertain: {pred} and {runner_up} score close to each other."

    notes = []
    if "Haemorrhagic" in class_names:
        haem = probs[class_names.index("Haemorrhagic")]
        if pred != "Haemorrhagic" and haem >= 0.15:
            notes.append(
                f"Haemorrhagic probability is {haem:.0%}. On the test set the model caught only "
                "half of the haemorrhagic scans, so this score is worth noting."
            )
    if pred == "Normal":
        notes.append(
            "A Normal prediction does not rule out stroke. Haemorrhagic recall on the test set was 50%, "
            "and the missed cases were predicted as Normal."
        )
    if pred == "Ischemic":
        notes.append(
            "Most ischemic training images come from ISLES-2022, so part of this signal may reflect "
            "the dataset source rather than pathology."
        )
    if pred == "Haemorrhagic":
        notes.append(
            "All haemorrhagic training images come from only 5 patients, so the model has seen "
            "limited variety for this class."
        )

    return {
        "level": level,
        "summary": summary,
        "top": float(top),
        "margin": float(margin),
        "notes": notes,
    }
