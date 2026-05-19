"""
RAG-based conversational medical assistant using Claude API.
Grounds all answers in the patient's own report + medical knowledge.
"""

import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def build_patient_context(
    clinical_text: str,
    entities: dict,
    summaries: dict,
    risk_result: dict,
    drug_result: dict,
    consistency_result: dict,
) -> dict:
    """
    Build a structured context object from all analysis results.
    This context is injected into every chatbot prompt as the knowledge base.
    """
    return {
        "clinical_text": clinical_text,
        "diseases": entities.get("diseases", []),
        "medications": entities.get("medications", []),
        "symptoms": entities.get("symptoms", []),
        "labs": entities.get("labs", []),
        "doctor_summary": summaries.get("doctor_summary", ""),
        "patient_summary": summaries.get("patient_summary", ""),
        "critical_alerts": summaries.get("critical_alerts", []),
        "readmission_risk": risk_result.get("readmission_risk", 0),
        "emergency_risk": risk_result.get("emergency_risk", 0),
        "severity": risk_result.get("severity", "UNKNOWN"),
        "drug_interactions": drug_result.get("interactions", []),
        "allergy_conflicts": drug_result.get("allergy_conflicts", []),
        "xray_findings": consistency_result.get("xray_findings", []),
        "missed_finding_alerts": consistency_result.get("missed_finding_alerts", []),
        "agreement_score": consistency_result.get("agreement_score", 1.0),
    }


def ask(question: str, patient_context: dict, chat_history: list[dict] | None = None) -> str:
    """
    Answer a clinical question grounded strictly in the patient's context.

    Args:
        question:        User's question (doctor or patient)
        patient_context: Output of build_patient_context()
        chat_history:    List of {"role": "user"/"assistant", "content": "..."} dicts

    Returns:
        Answer string
    """
    client = _get_client()
    system_prompt = _build_system_prompt(patient_context)

    messages = list(chat_history or [])
    messages.append({"role": "user", "content": question})

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=system_prompt,
        messages=messages,
    )

    return response.content[0].text.strip()


def _build_system_prompt(ctx: dict) -> str:
    drug_interactions_text = ""
    if ctx["drug_interactions"]:
        lines = [
            f"  • {i['drugs'][0]} + {i['drugs'][1]}: {i['severity']} — {i['effect']}"
            for i in ctx["drug_interactions"]
        ]
        drug_interactions_text = "Drug Interactions Detected:\n" + "\n".join(lines)
    else:
        drug_interactions_text = "Drug Interactions: None detected."

    missed_text = ""
    if ctx["missed_finding_alerts"]:
        lines = [f"  • {a['finding']} (confidence: {a['confidence']:.0%})" for a in ctx["missed_finding_alerts"]]
        missed_text = "Potential Missed X-ray Findings:\n" + "\n".join(lines)

    return f"""You are a clinical AI assistant helping patients and doctors understand a medical report.

PATIENT RECORD SUMMARY:
- Diagnosed Conditions: {', '.join(ctx['diseases']) or 'Not specified'}
- Current Medications: {', '.join(ctx['medications']) or 'Not specified'}
- Symptoms: {', '.join(ctx['symptoms']) or 'Not specified'}
- Lab Findings: {', '.join(ctx['labs']) or 'Not specified'}
- X-ray Findings: {', '.join(ctx['xray_findings']) or 'No imaging'}
- Readmission Risk: {ctx['readmission_risk']:.0%} ({ctx['severity']})
- Emergency Risk: {ctx['emergency_risk']:.0%}

{drug_interactions_text}
{missed_text}

CLINICAL SUMMARY:
{ctx['doctor_summary']}

FULL CLINICAL TEXT:
{ctx['clinical_text'][:3000]}

RULES:
1. Base all answers strictly on the patient record above.
2. If information is not in the record, say so clearly — do not hallucinate.
3. Explain medical terms simply when speaking to a patient.
4. Always recommend consulting a physician for treatment decisions.
5. For drug interaction questions, cite the specific drugs and severity.
6. For missed X-ray findings, clearly flag them as requiring physician review.
7. Keep answers concise (under 200 words) unless detail is explicitly requested."""
