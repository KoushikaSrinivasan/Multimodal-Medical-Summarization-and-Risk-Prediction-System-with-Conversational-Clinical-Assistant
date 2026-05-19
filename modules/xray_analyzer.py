"""
Chest X-ray analysis using torchxrayvision DenseNet (NIH pre-trained).
Returns per-pathology confidence scores and positive findings list.
"""

import numpy as np
from pathlib import Path
from typing import Any

try:
    import torchxrayvision as xrv
    import torch
    import skimage.io
    import skimage.transform
    _XRV_AVAILABLE = True
except ImportError:
    _XRV_AVAILABLE = False

from config import XRAY_MODEL, XRAY_CONFIDENCE_THRESHOLD, NIH_LABELS


_model = None


def _get_model():
    global _model
    if _model is None:
        if not _XRV_AVAILABLE:
            raise RuntimeError(
                "torchxrayvision is not installed. Run: pip install torchxrayvision"
            )
        _model = xrv.models.DenseNet(weights=XRAY_MODEL)
        _model.eval()
    return _model


def analyze_xray(image_path: str | Path) -> dict[str, Any]:
    """
    Classify a chest X-ray image and return findings.

    Returns:
        {
          "findings": ["Pneumonia", "Effusion"],       # above threshold
          "scores": {"Pneumonia": 0.82, ...},          # all 14 labels
          "top_finding": "Pneumonia",
          "confidence": 0.82,
          "raw_predictions": [...],
        }
    """
    if not _XRV_AVAILABLE:
        return _fallback_response()

    model = _get_model()
    img = _load_image(image_path)

    with torch.no_grad():
        output = model(img)

    predictions = torch.sigmoid(output).squeeze().numpy()

    scores: dict[str, float] = {}
    for label, score in zip(model.pathologies, predictions):
        if label in NIH_LABELS:
            scores[label] = float(score)

    findings = [label for label, score in scores.items() if score >= XRAY_CONFIDENCE_THRESHOLD]
    findings.sort(key=lambda x: scores[x], reverse=True)

    top_finding = findings[0] if findings else None
    top_confidence = scores[top_finding] if top_finding else 0.0

    return {
        "findings": findings,
        "scores": scores,
        "top_finding": top_finding,
        "confidence": top_confidence,
        "normal": len(findings) == 0,
    }


def _load_image(image_path: str | Path) -> "torch.Tensor":
    import torch
    img = skimage.io.imread(str(image_path))

    # Convert to grayscale if RGB
    if img.ndim == 3:
        img = img.mean(axis=2)

    # Resize to 224x224
    img = skimage.transform.resize(img, (224, 224), anti_aliasing=True)

    # Normalize to [-1024, 1024] range expected by torchxrayvision
    img = xrv.datasets.normalize(img.astype(np.float32), maxval=255, reshape=True)

    return torch.from_numpy(img).unsqueeze(0)  # (1, 1, 224, 224)


def _fallback_response() -> dict[str, Any]:
    """Used when torchxrayvision is not available (e.g., in testing)."""
    return {
        "findings": [],
        "scores": {label: 0.0 for label in NIH_LABELS},
        "top_finding": None,
        "confidence": 0.0,
        "normal": True,
        "warning": "torchxrayvision not available — install it to enable X-ray analysis.",
    }
