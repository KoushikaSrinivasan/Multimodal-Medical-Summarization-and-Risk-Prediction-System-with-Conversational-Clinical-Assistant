"""
NOVEL CONTRIBUTION: Cross-modal Clinical Consistency Scoring.

Detects mismatches between findings documented in clinical text (via NER)
and findings detected in the chest X-ray image (via CNN).

Clinical Insight: Radiological findings that are present in the X-ray but
absent from the clinical notes represent potential missed diagnoses or
documentation errors — a patient-safety concern rarely addressed by AI systems.

Algorithm:
  1. Map NER disease entities → NIH Chest X-ray label space (text_findings)
  2. Threshold X-ray CNN outputs → set of positive radiological findings (xray_findings)
  3. Compute:
       - agreed:    found in BOTH text and X-ray
       - text_only: documented in text but NOT seen in X-ray
       - xray_only: seen in X-ray but NOT documented in text  ← critical alerts
  4. Agreement score = |agreed| / |text_union_xray|  (Jaccard index)
  5. Each xray_only finding is a "potential missed finding" alert
"""

from config import TEXT_TO_XRAY_LABEL, XRAY_CONFIDENCE_THRESHOLD


def compute_consistency(
    entities: dict,
    xray_result: dict,
) -> dict:
    """
    Cross-validate clinical text findings against X-ray image findings.

    Args:
        entities:    Output of text_processor.extract_entities()
        xray_result: Output of xray_analyzer.analyze_xray()

    Returns:
        {
          "agreement_score": 0.67,           # Jaccard index [0, 1]
          "text_findings": [...],            # conditions mapped from NER
          "xray_findings": [...],            # positive X-ray labels
          "agreed": [...],                   # in both
          "text_only": [...],               # documented but not seen in X-ray
          "xray_only": [...],               # seen in X-ray but undocumented ← alerts
          "missed_finding_alerts": [...],    # human-readable alert strings
          "consistency_label": "HIGH",       # HIGH / MODERATE / LOW
        }
    """
    text_findings = _map_text_to_xray_labels(entities)
    xray_findings = set(xray_result.get("findings", []))

    agreed = text_findings & xray_findings
    text_only = text_findings - xray_findings
    xray_only = xray_findings - text_findings

    union = text_findings | xray_findings
    agreement_score = len(agreed) / len(union) if union else 1.0

    missed_alerts = _generate_missed_alerts(xray_only, xray_result.get("scores", {}))

    return {
        "agreement_score": round(agreement_score, 3),
        "text_findings": sorted(text_findings),
        "xray_findings": sorted(xray_findings),
        "agreed": sorted(agreed),
        "text_only": sorted(text_only),
        "xray_only": sorted(xray_only),
        "missed_finding_alerts": missed_alerts,
        "consistency_label": _label(agreement_score),
        "num_missed_findings": len(xray_only),
    }


def _map_text_to_xray_labels(entities: dict) -> set[str]:
    """
    Map disease/symptom NER entities to NIH Chest X-ray label space.
    Uses the TEXT_TO_XRAY_LABEL dictionary from config.py.
    """
    diseases = entities.get("diseases", []) + entities.get("symptoms", [])
    mapped: set[str] = set()

    for disease in diseases:
        normalized = disease.lower().strip()
        # Direct match
        if normalized in TEXT_TO_XRAY_LABEL:
            mapped.add(TEXT_TO_XRAY_LABEL[normalized])
            continue
        # Partial match — entity mentions any mapped keyword
        for keyword, xray_label in TEXT_TO_XRAY_LABEL.items():
            if keyword in normalized or normalized in keyword:
                mapped.add(xray_label)
                break

    return mapped


def _generate_missed_alerts(xray_only: set[str], scores: dict) -> list[dict]:
    """
    Generate human-readable alerts for findings in X-ray that are
    absent from clinical documentation.
    """
    alerts = []
    severity_map = {
        "Pneumothorax": "CRITICAL",
        "Cardiomegaly": "HIGH",
        "Edema": "HIGH",
        "Consolidation": "HIGH",
        "Pneumonia": "HIGH",
        "Effusion": "MODERATE",
        "Atelectasis": "MODERATE",
        "Mass": "HIGH",
        "Nodule": "MODERATE",
        "Infiltration": "MODERATE",
        "Emphysema": "LOW",
        "Fibrosis": "LOW",
        "Hernia": "LOW",
        "Pleural_Thickening": "LOW",
    }

    for finding in sorted(xray_only):
        confidence = scores.get(finding, 0.0)
        alert_severity = severity_map.get(finding, "MODERATE")
        alerts.append({
            "finding": finding,
            "confidence": round(confidence, 3),
            "alert_severity": alert_severity,
            "message": (
                f"[{alert_severity}] '{finding}' detected in X-ray "
                f"(confidence: {confidence:.0%}) but NOT mentioned in clinical notes. "
                f"Possible missed documentation or missed diagnosis."
            ),
        })

    # Sort by severity then confidence
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    alerts.sort(key=lambda a: (severity_order.get(a["alert_severity"], 9), -a["confidence"]))

    return alerts


def _label(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.40:
        return "MODERATE"
    return "LOW"
