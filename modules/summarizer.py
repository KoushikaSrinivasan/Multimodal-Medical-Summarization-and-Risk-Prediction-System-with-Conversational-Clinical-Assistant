"""
Dual-audience medical summarization using BART directly (no pipeline).
Compatible with transformers 5.x. Runs fully locally — no API key required.
"""

import re
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from config import SUMMARIZATION_MODEL

_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = BartTokenizer.from_pretrained(SUMMARIZATION_MODEL)
        _model = BartForConditionalGeneration.from_pretrained(SUMMARIZATION_MODEL)
        _model.eval()
    return _tokenizer, _model


def _summarize(text: str, max_length: int, min_length: int) -> str:
    tokenizer, model = _get_model()
    inputs = tokenizer(
        text[:3800],
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def generate_summaries(
    clinical_text: str,
    entities: dict,
    risk_score: float | None = None,
    xray_findings: list[str] | None = None,
) -> dict[str, str]:
    severity_label = _severity_label(risk_score)

    # Doctor summary — full clinical text
    doctor_summary = _summarize(clinical_text, max_length=280, min_length=120)

    # Patient summary — prepend simplified context
    patient_input = (
        f"Patient conditions: {', '.join(entities.get('diseases', [])) or 'not specified'}. "
        f"Medicines: {', '.join(entities.get('medications', [])) or 'none'}. "
        + clinical_text[:2000]
    )
    patient_summary = _summarize(patient_input, max_length=160, min_length=60)

    # Timeline
    timeline_input = (
        "Medical events: admission diagnosis treatment discharge followup. " + clinical_text[:2500]
    )
    timeline_raw = _summarize(timeline_input, max_length=120, min_length=40)
    timeline = _format_timeline(timeline_raw, clinical_text)

    critical_alerts = _extract_critical_alerts(entities, risk_score, xray_findings)

    return {
        "doctor_summary": doctor_summary,
        "patient_summary": patient_summary,
        "timeline": timeline,
        "critical_alerts": critical_alerts,
        "severity_label": severity_label,
    }


def _format_timeline(raw: str, clinical_text: str) -> str:
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", clinical_text)
    admission = dates[0] if len(dates) > 0 else "Not specified"
    discharge = dates[1] if len(dates) > 1 else "Not specified"
    return (
        f"• Admission  → {admission}\n"
        f"• Summary    → {raw}\n"
        f"• Discharge  → {discharge}\n"
        f"• Follow-up  → As per discharge instructions"
    )


def _severity_label(risk_score: float | None) -> str:
    if risk_score is None:
        return "UNKNOWN"
    if risk_score >= 0.7:
        return "CRITICAL"
    if risk_score >= 0.4:
        return "MEDIUM"
    return "LOW"


def _extract_critical_alerts(
    entities: dict,
    risk_score: float | None,
    xray_findings: list[str] | None,
) -> list[str]:
    alerts = []
    if risk_score is not None and risk_score >= 0.7:
        alerts.append(f"High readmission risk ({risk_score:.0%}) — close monitoring required.")
    critical_conditions = {
        "pneumothorax", "pulmonary embolism", "myocardial infarction",
        "stroke", "sepsis", "cardiac arrest", "heart failure",
    }
    for disease in entities.get("diseases", []):
        if disease.lower() in critical_conditions:
            alerts.append(f"Critical condition detected: {disease}")
    if xray_findings:
        urgent_xray = {"Pneumothorax", "Cardiomegaly", "Edema", "Consolidation"}
        for finding in xray_findings:
            if finding in urgent_xray:
                alerts.append(f"Urgent X-ray finding: {finding}")
    return alerts
