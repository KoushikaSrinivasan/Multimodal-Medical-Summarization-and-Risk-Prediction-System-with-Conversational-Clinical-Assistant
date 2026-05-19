"""
FastAPI backend — orchestrates all modules and exposes REST endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from modules.text_processor import extract_entities
from modules.summarizer import generate_summaries
from modules.xray_analyzer import analyze_xray
from modules.risk_predictor import predict_risk
from modules.drug_detector import detect_interactions
from modules.consistency_scorer import compute_consistency
from modules.chatbot import build_patient_context, ask
from modules.explainer import explain_risk


app = FastAPI(
    title="Medical AI System API",
    description="Multimodal Medical Summarization, Risk Prediction, and Clinical Assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (patient_context per session_id)
_sessions: dict[str, dict] = {}


class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    clinical_text: Annotated[str, Form()],
    patient_age: Annotated[int, Form()] = 65,
    num_prior_admissions: Annotated[int, Form()] = 1,
    time_in_hospital: Annotated[int, Form()] = 3,
    known_allergies: Annotated[str, Form()] = "",
    xray_image: UploadFile | None = File(default=None),
):
    """
    Full pipeline analysis endpoint.
    Accepts clinical text and optional X-ray image.
    Returns summaries, risk scores, drug interactions, and consistency report.
    """
    if not clinical_text.strip():
        raise HTTPException(status_code=422, detail="clinical_text cannot be empty.")

    # 1. NER
    entities = extract_entities(clinical_text)

    # 2. X-ray analysis (if image provided)
    xray_result = {"findings": [], "scores": {}, "top_finding": None, "confidence": 0.0, "normal": True}
    if xray_image and xray_image.filename:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(xray_image.filename).suffix) as tmp:
            tmp.write(await xray_image.read())
            tmp_path = tmp.name
        try:
            xray_result = analyze_xray(tmp_path)
        finally:
            os.unlink(tmp_path)

    # 3. Risk prediction
    patient_meta = {
        "age": patient_age,
        "num_prior_admissions": num_prior_admissions,
        "time_in_hospital": time_in_hospital,
    }
    risk_result = predict_risk(entities, patient_meta)

    # 4. Drug interaction detection
    allergies = [a.strip() for a in known_allergies.split(",") if a.strip()]
    drug_result = detect_interactions(entities.get("medications", []), allergies)

    # 5. Cross-modal consistency scoring (NOVEL)
    consistency_result = compute_consistency(entities, xray_result)

    # 6. Summarization (conditioned on risk score and X-ray findings)
    summaries = generate_summaries(
        clinical_text=clinical_text,
        entities=entities,
        risk_score=risk_result["readmission_risk"],
        xray_findings=xray_result.get("findings"),
    )

    # 7. SHAP explainability
    explanation = explain_risk(model=None, features=risk_result["features"])

    # Build and store patient context for chatbot
    session_id = _make_session_id(clinical_text)
    _sessions[session_id] = build_patient_context(
        clinical_text=clinical_text,
        entities=entities,
        summaries=summaries,
        risk_result=risk_result,
        drug_result=drug_result,
        consistency_result=consistency_result,
    )

    return {
        "session_id": session_id,
        "entities": entities,
        "summaries": summaries,
        "xray_result": xray_result,
        "risk_result": risk_result,
        "drug_result": drug_result,
        "consistency_result": consistency_result,
        "explanation": explanation,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Conversational endpoint — answers questions grounded in patient record."""
    context = _sessions.get(req.session_id)
    if context is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Run /analyze first to create a session.",
        )
    answer = ask(req.question, context, req.history)
    return ChatResponse(answer=answer, session_id=req.session_id)


def _make_session_id(text: str) -> str:
    import hashlib
    return hashlib.md5(text[:200].encode()).hexdigest()[:12]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
