"""
RAG-based conversational medical assistant using Flan-T5 (google/flan-t5-base).
Runs fully locally — no API key required.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import QA_MODEL

_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL)
        _model.eval()
    return _tokenizer, _model


def build_patient_context(
    clinical_text: str,
    entities: dict,
    summaries: dict,
    risk_result: dict,
    drug_result: dict,
    consistency_result: dict,
) -> dict:
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
    Answer a clinical question grounded in the patient context using Flan-T5 locally.
    """
    tokenizer, model = _get_model()

    # Build compact context (Flan-T5 base has 512 token limit)
    context = _build_compact_context(patient_context)

    prompt = (
        f"You are a medical assistant. Answer the question using only the patient record below.\n\n"
        f"Patient Record:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Fallback if model returns empty or very short answer
    if len(answer) < 10:
        answer = _rule_based_answer(question, patient_context)

    return answer


def _build_compact_context(ctx: dict) -> str:
    """Compress patient context to fit within Flan-T5's 512 token limit."""
    drug_info = ""
    if ctx["drug_interactions"]:
        pairs = [f"{i['drugs'][0]}+{i['drugs'][1]}({i['severity']})" for i in ctx["drug_interactions"][:3]]
        drug_info = f"Drug interactions: {', '.join(pairs)}."

    missed = ""
    if ctx["missed_finding_alerts"]:
        findings = [a["finding"] for a in ctx["missed_finding_alerts"][:2]]
        missed = f"Missed X-ray findings: {', '.join(findings)}."

    return (
        f"Diagnoses: {', '.join(ctx['diseases'][:5]) or 'unknown'}. "
        f"Medications: {', '.join(ctx['medications'][:6]) or 'none'}. "
        f"Symptoms: {', '.join(ctx['symptoms'][:4]) or 'none'}. "
        f"Labs: {', '.join(ctx['labs'][:4]) or 'none'}. "
        f"X-ray: {', '.join(ctx['xray_findings'][:3]) or 'none'}. "
        f"Readmission risk: {ctx['readmission_risk']:.0%} ({ctx['severity']}). "
        f"Emergency risk: {ctx['emergency_risk']:.0%}. "
        f"{drug_info} {missed} "
        f"Summary: {ctx['doctor_summary'][:300]}"
    )


def _rule_based_answer(question: str, ctx: dict) -> str:
    """Fallback answers for common questions when model output is poor."""
    q = question.lower()

    if any(k in q for k in ["diagnos", "disease", "condition", "problem"]):
        diseases = ', '.join(ctx['diseases']) or "Not identified"
        return f"The patient's diagnosed conditions are: {diseases}."

    if any(k in q for k in ["medic", "drug", "tablet", "pill", "prescription"]):
        meds = ', '.join(ctx['medications']) or "None listed"
        return f"The patient's medications include: {meds}."

    if any(k in q for k in ["risk", "readmit", "danger"]):
        return (
            f"The patient has a readmission risk of {ctx['readmission_risk']:.0%} "
            f"({ctx['severity']} severity) and an emergency risk of {ctx['emergency_risk']:.0%}."
        )

    if any(k in q for k in ["interaction", "conflict", "safe"]):
        if ctx["drug_interactions"]:
            pairs = [f"{i['drugs'][0]} and {i['drugs'][1]} ({i['severity']})" for i in ctx["drug_interactions"]]
            return f"Drug interactions detected: {'; '.join(pairs)}."
        return "No drug interactions were detected for this patient."

    if any(k in q for k in ["xray", "x-ray", "scan", "image", "finding"]):
        findings = ', '.join(ctx['xray_findings']) or "No imaging provided"
        return f"X-ray findings: {findings}."

    if any(k in q for k in ["missed", "miss"]):
        if ctx["missed_finding_alerts"]:
            missed = [a["finding"] for a in ctx["missed_finding_alerts"]]
            return f"Potential missed findings in X-ray (not documented in notes): {', '.join(missed)}."
        return "No missed findings detected — X-ray and clinical notes are consistent."

    return (
        f"Based on the patient record: diagnoses are {', '.join(ctx['diseases'][:3]) or 'unknown'}, "
        f"medications include {', '.join(ctx['medications'][:3]) or 'none'}, "
        f"and readmission risk is {ctx['readmission_risk']:.0%}. "
        f"Please consult a physician for detailed medical advice."
    )
