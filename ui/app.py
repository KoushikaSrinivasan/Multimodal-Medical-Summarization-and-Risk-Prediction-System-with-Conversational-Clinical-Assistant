"""
Streamlit frontend for the Multimodal Medical AI System.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from modules.text_processor import extract_entities, extract_patient_meta
from modules.summarizer import generate_summaries
from modules.xray_analyzer import analyze_xray
from modules.risk_predictor import predict_risk
from modules.drug_detector import detect_interactions
from modules.consistency_scorer import compute_consistency
from modules.chatbot import build_patient_context, ask
from modules.explainer import explain_risk


st.set_page_config(
    page_title="Medical AI System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.severity-critical { color: #d32f2f; font-weight: bold; font-size: 1.1em; }
.severity-medium   { color: #f57c00; font-weight: bold; font-size: 1.1em; }
.severity-low      { color: #388e3c; font-weight: bold; font-size: 1.1em; }
.alert-box  { background: #fff3e0; border-left: 4px solid #ff9800; padding: 10px; border-radius: 4px; margin: 6px 0; }
.missed-box { background: #fce4ec; border-left: 4px solid #e91e63; padding: 10px; border-radius: 4px; margin: 6px 0; }
.agreed-box { background: #e8f5e9; border-left: 4px solid #4caf50; padding: 10px; border-radius: 4px; margin: 6px 0; }
</style>
""", unsafe_allow_html=True)


# ─── Session state defaults ──────────────────────────────────────────────────────
_defaults = {
    "analysis": None,
    "patient_context": None,
    "chat_history": [],
    # Backing variables for sidebar widgets (not bound as widget keys)
    "meta_age": 65,
    "meta_admissions": 1,
    "meta_stay": 3,
    "auto_detected": False,
    # Two-phase analysis: store pending inputs, rerun, then execute
    "pending_analysis": False,
    "pending_text": "",
    "pending_xray_bytes": None,
    "pending_xray_name": None,
    "pending_allergies": "",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── PHASE 2: Execute pending analysis (runs after rerun updates sidebar) ────────
if st.session_state.pending_analysis and st.session_state.pending_text:
    st.session_state.pending_analysis = False
    _text      = st.session_state.pending_text
    _xray_b    = st.session_state.pending_xray_bytes
    _xray_name = st.session_state.pending_xray_name
    _allergies = st.session_state.pending_allergies

    with st.spinner("Running multimodal analysis... (first run downloads models ~2 GB)"):

        entities = extract_entities(_text)

        xray_result = {"findings": [], "scores": {}, "top_finding": None, "confidence": 0.0, "normal": True}
        if _xray_b:
            suffix = Path(_xray_name).suffix if _xray_name else ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(_xray_b)
                tmp_path = tmp.name
            try:
                xray_result = analyze_xray(tmp_path)
            finally:
                os.unlink(tmp_path)

        patient_meta = {
            "age": st.session_state.meta_age,
            "num_prior_admissions": st.session_state.meta_admissions,
            "time_in_hospital": st.session_state.meta_stay,
        }
        risk_result = predict_risk(entities, patient_meta)

        allergy_list = [a.strip() for a in _allergies.split(",") if a.strip()]
        drug_result = detect_interactions(entities.get("medications", []), allergy_list)

        consistency_result = compute_consistency(entities, xray_result)

        summaries = generate_summaries(
            clinical_text=_text,
            entities=entities,
            risk_score=risk_result["readmission_risk"],
            xray_findings=xray_result.get("findings"),
        )

        explanation = explain_risk(model=None, features=risk_result["features"])

        st.session_state.analysis = {
            "entities": entities,
            "xray_result": xray_result,
            "risk_result": risk_result,
            "drug_result": drug_result,
            "consistency_result": consistency_result,
            "summaries": summaries,
            "explanation": explanation,
        }
        st.session_state.patient_context = build_patient_context(
            clinical_text=_text,
            entities=entities,
            summaries=summaries,
            risk_result=risk_result,
            drug_result=drug_result,
            consistency_result=consistency_result,
        )
        st.session_state.chat_history = []

    st.success("Analysis complete. View results in the tabs above.")


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Medical AI System")
    st.caption("Multimodal Summarization · Risk Prediction · Clinical Assistant")
    st.divider()

    st.subheader("Patient Information")
    if st.session_state.auto_detected:
        st.success("Auto-detected from report")

    # Use value= (no key=) so backing vars can be updated freely between reruns
    patient_age          = st.slider("Patient Age", 1, 120, value=int(st.session_state.meta_age))
    num_prior_admissions = st.number_input("Prior Hospital Admissions", 0, 20, value=int(st.session_state.meta_admissions))
    time_in_hospital     = st.number_input("Current Stay (days)", 1, 180, value=int(st.session_state.meta_stay))
    known_allergies      = st.text_input("Known Allergies (comma-separated)", placeholder="e.g. penicillin, nsaid")

    st.divider()
    st.caption("Powered by BART · Flan-T5 · BioBERT · torchxrayvision · XGBoost · SHAP")


# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab_input, tab_summary, tab_risk, tab_xray, tab_drugs, tab_consistency, tab_chat = st.tabs([
    "📄 Input", "📋 Summaries", "⚠️ Risk", "🔬 X-Ray", "💊 Drugs", "🔗 Consistency", "💬 Assistant",
])


# ─── TAB 1: Input ────────────────────────────────────────────────────────────────
with tab_input:
    st.header("Upload Medical Records")

    col1, col2 = st.columns([2, 1])
    with col1:
        clinical_text = st.text_area(
            "Clinical Text (discharge note / lab report / prescription)",
            height=320,
            placeholder="Paste discharge summary, lab report, or clinical notes here...",
        )
    with col2:
        xray_file = st.file_uploader("Chest X-Ray Image (optional)", type=["png", "jpg", "jpeg"])
        if xray_file:
            st.image(xray_file, caption="Uploaded X-Ray", use_column_width=True)

    if st.button("Analyze", type="primary", use_container_width=True):
        if not clinical_text.strip():
            st.error("Please enter clinical text to analyze.")
        else:
            # ── PHASE 1: Auto-detect meta, store inputs, trigger rerun ──
            auto_meta = extract_patient_meta(clinical_text)
            if auto_meta:
                st.session_state.meta_age        = auto_meta.get("age", st.session_state.meta_age)
                st.session_state.meta_admissions = auto_meta.get("num_prior_admissions", st.session_state.meta_admissions)
                st.session_state.meta_stay       = auto_meta.get("time_in_hospital", st.session_state.meta_stay)
                st.session_state.auto_detected   = True
            else:
                # Use current manual sidebar values
                st.session_state.meta_age        = patient_age
                st.session_state.meta_admissions = num_prior_admissions
                st.session_state.meta_stay       = time_in_hospital
                st.session_state.auto_detected   = False

            # Store inputs for Phase 2
            st.session_state.pending_text       = clinical_text
            st.session_state.pending_xray_bytes = xray_file.read() if xray_file else None
            st.session_state.pending_xray_name  = xray_file.name if xray_file else None
            st.session_state.pending_allergies  = known_allergies
            st.session_state.pending_analysis   = True
            st.rerun()


# ─── TAB 2: Summaries ────────────────────────────────────────────────────────────
with tab_summary:
    if st.session_state.analysis is None:
        st.info("Run analysis first from the Input tab.")
    else:
        a        = st.session_state.analysis
        summaries = a["summaries"]
        entities  = a["entities"]

        if summaries.get("critical_alerts"):
            st.error("CRITICAL ALERTS")
            for alert in summaries["critical_alerts"]:
                st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Clinical Summary (Doctor)")
            st.write(summaries["doctor_summary"])
        with col2:
            st.subheader("Patient-Friendly Summary")
            st.write(summaries["patient_summary"])

        st.subheader("Medical Timeline")
        st.write(summaries["timeline"])


# ─── TAB 3: Risk ─────────────────────────────────────────────────────────────────
with tab_risk:
    if st.session_state.analysis is None:
        st.info("Run analysis first from the Input tab.")
    else:
        risk = st.session_state.analysis["risk_result"]
        expl = st.session_state.analysis["explanation"]
        severity = risk["severity"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Readmission Risk", f"{risk['readmission_risk']:.0%}")
        col2.metric("Emergency Risk",   f"{risk['emergency_risk']:.0%}")
        col3.markdown(f'<p class="severity-{severity.lower()}">Severity: {severity}</p>', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk["readmission_risk"] * 100,
            title={"text": "Readmission Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d32f2f" if risk["readmission_risk"] >= 0.7 else
                                 "#f57c00" if risk["readmission_risk"] >= 0.4 else "#388e3c"},
                "steps": [
                    {"range": [0, 40],  "color": "#e8f5e9"},
                    {"range": [40, 70], "color": "#fff3e0"},
                    {"range": [70, 100],"color": "#fce4ec"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 70},
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Why This Risk Score? (SHAP Explanation)")
        st.text(expl["explanation_text"])

        if expl["top_factors"]:
            factors = expl["top_factors"]
            fig2 = px.bar(
                x=[f["shap_value"] for f in factors],
                y=[f["label"] for f in factors],
                orientation="h",
                color=[f["direction"] for f in factors],
                color_discrete_map={"increases": "#d32f2f", "decreases": "#388e3c"},
                labels={"x": "SHAP Value (impact on risk)", "y": "Feature"},
                title="Feature Impact on Readmission Risk",
            )
            fig2.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)


# ─── TAB 4: X-Ray ────────────────────────────────────────────────────────────────
with tab_xray:
    if st.session_state.analysis is None:
        st.info("Run analysis first from the Input tab.")
    else:
        xray = st.session_state.analysis["xray_result"]

        if xray.get("warning"):
            st.warning(xray["warning"])
        elif xray.get("normal"):
            st.success("No significant findings detected in X-ray.")
        else:
            st.subheader(f"Top Finding: {xray['top_finding']} ({xray['confidence']:.0%} confidence)")
            for finding in xray["findings"]:
                st.progress(xray["scores"].get(finding, 0), text=f"{finding}: {xray['scores'].get(finding, 0):.0%}")

        if xray.get("scores"):
            scores = xray["scores"]
            fig = px.bar(
                x=list(scores.values()), y=list(scores.keys()),
                orientation="h", color=list(scores.values()),
                color_continuous_scale=["#e8f5e9", "#fff3e0", "#d32f2f"],
                title="X-Ray Pathology Confidence Scores (NIH 14-class)",
                labels={"x": "Confidence", "y": "Pathology"},
            )
            fig.add_vline(x=0.3, line_dash="dash", annotation_text="Detection threshold (0.3)")
            fig.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ─── TAB 5: Drugs ────────────────────────────────────────────────────────────────
with tab_drugs:
    if st.session_state.analysis is None:
        st.info("Run analysis first from the Input tab.")
    else:
        drug = st.session_state.analysis["drug_result"]

        if drug["safe"]:
            st.success(f"No drug interactions detected among {drug['total_medications']} medications.")
        else:
            st.error(f"Drug safety issues detected! Highest severity: {drug['highest_severity']}")

        if drug["interactions"]:
            st.subheader("Drug-Drug Interactions")
            for interaction in drug["interactions"]:
                icon = {"CONTRAINDICATED": "🔴", "MAJOR": "🟠", "MODERATE": "🟡"}.get(interaction["severity"], "🟡")
                with st.expander(f"{icon} {interaction['drugs'][0]} + {interaction['drugs'][1]} — {interaction['severity']}"):
                    st.write(f"**Effect:** {interaction['effect']}")
                    st.write(f"**Recommendation:** {interaction['recommendation']}")

        if drug["allergy_conflicts"]:
            st.subheader("Allergy Conflicts")
            for conflict in drug["allergy_conflicts"]:
                st.markdown(
                    f'<div class="alert-box">🚨 <b>{conflict["medication"]}</b> conflicts with '
                    f'<b>{conflict["conflict_class"]}</b> allergy — {conflict["note"]}</div>',
                    unsafe_allow_html=True,
                )


# ─── TAB 6: Consistency (NOVEL) ──────────────────────────────────────────────────
with tab_consistency:
    if st.session_state.analysis is None:
        st.info("Run analysis first from the Input tab.")
    else:
        cons = st.session_state.analysis["consistency_result"]

        st.header("Cross-Modal Clinical Consistency Analysis")
        st.caption(
            "Compares findings in clinical text (NER) vs X-ray image (CNN). "
            "Mismatches may indicate missed diagnoses or documentation errors."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Agreement Score",        f"{cons['agreement_score']:.0%}")
        col2.metric("Consistency Level",      cons["consistency_label"])
        col3.metric("Potential Missed Findings", cons["num_missed_findings"])

        vcol1, vcol2, vcol3 = st.columns(3)
        with vcol1:
            st.subheader("Agreed (Both)")
            for f in cons["agreed"]:
                st.markdown(f'<div class="agreed-box">✓ {f}</div>', unsafe_allow_html=True)
            if not cons["agreed"]:
                st.write("None")
        with vcol2:
            st.subheader("Text Only")
            for f in cons["text_only"]:
                st.markdown(f'<div class="alert-box">📄 {f}</div>', unsafe_allow_html=True)
            if not cons["text_only"]:
                st.write("None")
        with vcol3:
            st.subheader("X-Ray Only (Missed?)")
            for f in cons["xray_only"]:
                st.markdown(f'<div class="missed-box">🔬 {f}</div>', unsafe_allow_html=True)
            if not cons["xray_only"]:
                st.write("None")

        if cons["missed_finding_alerts"]:
            st.subheader("Missed Finding Alerts")
            for alert in cons["missed_finding_alerts"]:
                emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MODERATE": "⚡", "LOW": "ℹ️"}.get(alert["alert_severity"], "⚡")
                st.markdown(f'<div class="missed-box">{emoji} {alert["message"]}</div>', unsafe_allow_html=True)

        if cons["xray_findings"] or cons["text_findings"]:
            all_f    = sorted(set(cons["xray_findings"]) | set(cons["text_findings"]))
            text_v   = [1 if f in cons["text_findings"] else 0 for f in all_f]
            xray_v   = [1 if f in cons["xray_findings"] else 0 for f in all_f]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=text_v + [text_v[0]], theta=all_f + [all_f[0]],
                                          fill="toself", name="Clinical Text", line_color="#1565C0"))
            fig.add_trace(go.Scatterpolar(r=xray_v + [xray_v[0]], theta=all_f + [all_f[0]],
                                          fill="toself", name="X-Ray CNN", line_color="#b71c1c", opacity=0.6))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                              showlegend=True, title="Text vs X-Ray Findings Overlap", height=400)
            st.plotly_chart(fig, use_container_width=True)


# ─── TAB 7: Chatbot ──────────────────────────────────────────────────────────────
with tab_chat:
    st.header("Clinical Assistant")
    st.caption("Ask questions about the patient's report. Grounded entirely in the analyzed record.")

    if st.session_state.patient_context is None:
        st.info("Run analysis first from the Input tab to enable the assistant.")
    else:
        st.subheader("Suggested Questions")
        suggestions = [
            "What is the patient's primary diagnosis?",
            "Why was this medication prescribed?",
            "What are the most dangerous drug interactions?",
            "What caused the high readmission risk?",
            "What follow-up care is needed?",
            "Are there any missed findings in the X-ray?",
        ]
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            if cols[i % 3].button(suggestion, key=f"sug_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": suggestion})
                with st.spinner("Thinking..."):
                    response = ask(suggestion, st.session_state.patient_context, st.session_state.chat_history[:-1])
                st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.divider()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask a question about this patient's record...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("..."):
                    response = ask(user_input, st.session_state.patient_context, st.session_state.chat_history[:-1])
                st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        if st.session_state.chat_history:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
