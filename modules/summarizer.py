"""
Dual-audience medical summarization using Claude API.
Generates a doctor-facing clinical summary and a patient-friendly summary.
"""

import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def generate_summaries(
    clinical_text: str,
    entities: dict,
    risk_score: float | None = None,
    xray_findings: list[str] | None = None,
) -> dict[str, str]:
    """
    Generate doctor summary, patient summary, and critical alerts.
    risk_score (0-1) is used to condition summary emphasis.
    Returns dict with keys: doctor_summary, patient_summary, critical_alerts, timeline.
    """
    client = _get_client()

    severity_label = _severity_label(risk_score)
    xray_context = (
        f"X-ray findings: {', '.join(xray_findings)}" if xray_findings else "No imaging provided."
    )

    system_prompt = (
        "You are a clinical AI assistant that generates accurate, structured medical summaries. "
        "Always be precise, factual, and base your output strictly on the provided clinical information."
    )

    doctor_prompt = f"""Generate a structured clinical summary for a physician.

CLINICAL TEXT:
{clinical_text}

EXTRACTED ENTITIES:
- Diseases/Conditions: {', '.join(entities.get('diseases', [])) or 'None identified'}
- Medications: {', '.join(entities.get('medications', [])) or 'None identified'}
- Symptoms: {', '.join(entities.get('symptoms', [])) or 'None identified'}
- Lab findings: {', '.join(entities.get('labs', [])) or 'None identified'}

{xray_context}
OVERALL RISK LEVEL: {severity_label}

Write a concise clinical summary (200-300 words) covering:
1. Chief complaint and primary diagnosis
2. Key clinical findings and lab results
3. Treatment administered
4. Risk factors and comorbidities
5. Recommended follow-up actions
{"6. CRITICAL ALERTS: Highlight urgent concerns prominently." if severity_label == "CRITICAL" else ""}"""

    patient_prompt = f"""Generate a simple, patient-friendly summary of the medical report below.

CLINICAL TEXT:
{clinical_text}

EXTRACTED CONDITIONS: {', '.join(entities.get('diseases', [])) or 'Not specified'}
MEDICATIONS: {', '.join(entities.get('medications', [])) or 'Not specified'}

Write in plain English (8th-grade reading level, 150-200 words):
1. What is wrong (diagnosis in simple terms)
2. What was done to treat it
3. What medicines were prescribed and why
4. What to watch out for at home
5. When to come back or seek emergency help"""

    timeline_prompt = f"""Create a brief medical timeline from this clinical text:

{clinical_text}

Format as a short bullet list with dates/stages:
• Admission → [date/event]
• Diagnosis → [what was found]
• Treatment → [what was done]
• Current status → [discharge / ongoing]
• Follow-up → [next steps]"""

    doctor_resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=system_prompt,
        messages=[{"role": "user", "content": doctor_prompt}],
    )

    patient_resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        system=system_prompt,
        messages=[{"role": "user", "content": patient_prompt}],
    )

    timeline_resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=250,
        system=system_prompt,
        messages=[{"role": "user", "content": timeline_prompt}],
    )

    critical_alerts = _extract_critical_alerts(entities, risk_score, xray_findings)

    return {
        "doctor_summary": doctor_resp.content[0].text.strip(),
        "patient_summary": patient_resp.content[0].text.strip(),
        "timeline": timeline_resp.content[0].text.strip(),
        "critical_alerts": critical_alerts,
        "severity_label": severity_label,
    }


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
