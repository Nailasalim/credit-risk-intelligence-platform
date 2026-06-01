"""
Explainability page — enterprise credit-risk explanation dashboard.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app_navigation import EXPLAIN_MODE_CUSTOM, EXPLAIN_MODE_LATEST

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import preprocess_applicant  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
SHAP_VALUES_PATH = MODELS_DIR / "shap_values.npy"
MODEL_PATH = MODELS_DIR / "model.pkl"

FEATURE_LABELS: dict[str, str] = {
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Credit amount",
    "AMT_ANNUITY": "Monthly annuity",
    "AMT_GOODS_PRICE": "Goods price",
    "DAYS_BIRTH": "Applicant age",
    "DAYS_EMPLOYED": "Employment history",
    "REGION_RATING_CLIENT": "Region rating",
    "REGION_RATING_CLIENT_W_CITY": "Region & city rating",
    "INCOME_CREDIT_RATIO": "Income-to-credit ratio",
    "ANNUITY_INCOME_RATIO": "Annuity-to-income ratio",
    "CREDIT_GOODS_RATIO": "Credit-to-goods ratio",
    "DAYS_LAST_PHONE_CHANGE": "Phone stability",
    "DAYS_ID_PUBLISH": "ID tenure",
    "REG_CITY_NOT_WORK_CITY": "Work vs registered address",
    "REG_CITY_NOT_LIVE_CITY": "Residence vs registered address",
    "FLAG_EMP_PHONE": "Employer phone",
    "FLAG_DOCUMENT_3": "ID document",
    "OWN_CAR_AGE": "Vehicle age",
}

BAND_COLORS = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
RECOMMENDATION_LABEL = {"APPROVE": "Approve", "REVIEW": "Manual review", "DECLINE": "Decline"}


# ---------------------------------------------------------------------------
# Data (unchanged logic)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model() -> Any:
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_names() -> list[str]:
    with FEATURE_NAMES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_global_importance() -> pd.DataFrame:
    names = load_feature_names()

    if SHAP_VALUES_PATH.is_file():
        shap_matrix = np.load(SHAP_VALUES_PATH)
        if shap_matrix.ndim != 2 or shap_matrix.shape[1] != len(names):
            raise ValueError(
                f"Expected shap_values shape (n_samples, {len(names)}), got {shap_matrix.shape}"
            )
        mean_abs = np.abs(shap_matrix).mean(axis=0)
        median_signed = np.median(shap_matrix, axis=0)
        direction = np.sign(median_signed)
        direction = np.where(direction == 0, np.sign(shap_matrix.mean(axis=0)), direction)
        direction = np.where(direction == 0, 1.0, direction)
        display_impact = direction * mean_abs
        source = "global_shap"
    else:
        model = load_model()
        raw = np.asarray(model.feature_importances_, dtype=float)
        mean_abs = raw / raw.sum()
        display_impact = mean_abs
        source = "model_gain"

    frame = pd.DataFrame(
        {
            "feature": names,
            "importance": mean_abs,
            "display_impact": display_impact,
            "source": source,
        }
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_contributions(applicant_payload: dict[str, Any]) -> pd.DataFrame:
    model = load_model()
    features = preprocess_applicant(applicant_payload)
    raw = model.predict(features, pred_contrib=True)[0]
    names = load_feature_names()
    values = raw[: len(names)]
    frame = pd.DataFrame({"feature": names, "impact": values})
    frame["abs_impact"] = frame["impact"].abs()
    return frame.sort_values("abs_impact", ascending=False).reset_index(drop=True)


def _concise_driver_title(feature: str, impact: float, payload: dict[str, Any]) -> str:
    val = payload.get(feature)
    if feature == "EXT_SOURCE_1":
        return "Low external credit score 1" if val is not None and float(val) < 0.5 else "External credit score 1"
    if feature == "EXT_SOURCE_2":
        return "Low EXT_SOURCE_2" if val is not None and float(val) < 0.5 else "EXT_SOURCE_2"
    if feature == "EXT_SOURCE_3":
        return "Low external credit score 3" if val is not None and float(val) < 0.5 else "External credit score 3"
    if feature == "DAYS_EMPLOYED" and val is not None and abs(int(val)) / 365.25 < 2:
        return "Short employment history"
    if feature == "ANNUITY_INCOME_RATIO" and val is not None and float(val) > 0.30:
        return "High annuity burden"
    if feature == "AMT_ANNUITY" and impact > 0:
        return "High annuity burden"
    if feature == "INCOME_CREDIT_RATIO" and val is not None and float(val) < 0.2:
        return "High credit burden"
    if feature == "AMT_INCOME_TOTAL" and impact < 0:
        return "Stable income"
    if feature == "FLAG_DOCUMENT_3" and val == 1:
        return "ID document verified"
    if feature == "FLAG_EMP_PHONE" and val == 1:
        return "Employer phone available"
    if feature == "OWN_CAR_AGE" and impact < 0:
        return "Vehicle ownership"
    if feature.startswith("EXT_SOURCE") and val is not None and float(val) >= 0.65:
        return "Strong credit history"
    if impact > 0:
        return FEATURE_LABELS.get(feature, feature)
    return FEATURE_LABELS.get(feature, feature)


def split_drivers(contributions: pd.DataFrame, payload: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    risk_rows = contributions[contributions["impact"] > 0].head(3)
    positive_rows = contributions[contributions["impact"] < 0].head(3)

    def pack(rows: pd.DataFrame) -> list[dict]:
        return [
            {
                "title": _concise_driver_title(row.feature, float(row.impact), payload),
                "impact": float(row.impact),
            }
            for row in rows.itertuples()
        ]

    return pack(risk_rows), pack(positive_rows)


def build_business_summary(
    risk_band: str,
    risk_drivers: list[dict],
    positive_factors: list[dict],
) -> str:
    band = {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"}.get(risk_band.upper(), risk_band)
    if not risk_drivers:
        return f"This applicant was classified as **{band} Risk** with a broadly balanced profile."
    drivers = ", ".join(d["title"].lower() for d in risk_drivers[:3])
    text = f"This applicant was classified as **{band} Risk** primarily due to {drivers}."
    if positive_factors:
        text += f" Partially offset by {', '.join(p['title'].lower() for p in positive_factors[:2])}."
    return text


def _has_latest_assessment() -> bool:
    return (
        st.session_state.get("last_applicant_payload") is not None
        and st.session_state.get("risk_result") is not None
    )


def _init_explainability_mode() -> None:
    if st.session_state.get("explainability_open_latest"):
        st.session_state.explainability_mode = EXPLAIN_MODE_LATEST
        st.session_state.pop("explainability_open_latest", None)
    elif "explainability_mode" not in st.session_state:
        st.session_state.explainability_mode = (
            EXPLAIN_MODE_LATEST if _has_latest_assessment() else EXPLAIN_MODE_CUSTOM
        )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def inject_explainability_styles() -> None:
    st.markdown(
        """
        <style>
        .xai-card {
            background: #161d27; border: 1px solid #243044;
            border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 0.6rem;
        }
        .xai-card-header {
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 0.65rem; gap: 0.5rem;
        }
        .xai-card-title {
            font-size: 0.88rem; font-weight: 600; color: #e8edf5; margin: 0;
        }
        .xai-badge {
            font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.06em; padding: 0.2rem 0.5rem; border-radius: 4px;
            background: rgba(59,130,246,0.15); color: #60a5fa; white-space: nowrap;
        }
        .xai-fi-head {
            display: flex; justify-content: space-between; color: #6b7c96;
            font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
            font-size: 0.62rem; padding-bottom: 0.35rem; margin-bottom: 0.15rem;
            border-bottom: 1px solid #243044;
        }
        .xai-fi-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.28rem 0; color: #c5d0e0; font-size: 0.78rem;
        }
        .xai-fi-row span:first-child {
            font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.72rem;
        }
        .xai-fi-legend {
            font-size: 0.64rem; color: #6b7c96; margin-top: 0.5rem;
            display: flex; gap: 0.75rem; flex-wrap: wrap;
        }
        .xai-fi-legend i {
            display: inline-block; width: 7px; height: 7px;
            border-radius: 2px; margin-right: 0.2rem;
        }
        .xai-fi-row .impact-up { color: #f87171; font-weight: 600; }
        .xai-fi-row .impact-down { color: #4ade80; font-weight: 600; }
        .xai-assess-title { font-size: 1rem; font-weight: 700; margin: 0; }
        .xai-assess-sub { font-size: 0.76rem; color: #94a3b8; margin: 0.25rem 0 0 0; }
        .xai-summary-body {
            font-size: 0.84rem; color: #c5d0e0; line-height: 1.55; margin: 0;
        }
        .xai-bullet {
            font-size: 0.78rem; color: #d1dae8; padding: 0.3rem 0.5rem;
            margin: 0 0 0.22rem 0; border-radius: 6px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .xai-bullet.risk { background: rgba(239,68,68,0.08); border-left: 2px solid #ef4444; }
        .xai-bullet.pos { background: rgba(34,197,94,0.08); border-left: 2px solid #22c55e; }
        .xai-bullet-impact { font-size: 0.7rem; font-weight: 600; }
        .xai-bullet-impact.up { color: #f87171; }
        .xai-bullet-impact.down { color: #4ade80; }
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 0.35rem;
        }
        div[data-testid="stRadio"] label {
            background: #121820; border: 1px solid #243044;
            border-radius: 8px; padding: 0.35rem 0.65rem !important;
            font-size: 0.78rem;
        }
        div[data-testid="stRadio"] label[data-checked="true"] {
            border-color: #3b82f6; background: rgba(59,130,246,0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_global_feature_importance_card() -> None:
    importance = load_global_importance()
    top10 = importance.head(10)
    data_source = str(top10["source"].iloc[0]) if not top10.empty else "global_shap"
    badge = "Portfolio SHAP" if data_source == "global_shap" else "Model gain"

    rows = []
    for row in top10.itertuples():
        impact = float(row.display_impact)
        css = "impact-up" if impact > 0 else "impact-down"
        sign = "+" if impact >= 0 else ""
        rows.append(
            f'<div class="xai-fi-row"><span>{row.feature}</span>'
            f'<span class="{css}">{sign}{impact:.2f}</span></div>'
        )

    st.markdown(
        f"""
        <div class="xai-card">
            <div class="xai-card-header">
                <p class="xai-card-title">Global Feature Importance</p>
                <span class="xai-badge">{badge}</span>
            </div>
            <div class="xai-fi-head"><span>Feature</span><span>Impact</span></div>
            {''.join(rows)}
            <div class="xai-fi-legend">
                <span><i style="background:#f87171;"></i>Increases default risk</span>
                <span><i style="background:#4ade80;"></i>Lowers default risk</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assessment_summary_card(decision: dict[str, Any]) -> None:
    band = str(decision.get("risk_band", "MEDIUM")).upper()
    color = BAND_COLORS.get(band, "#f59e0b")
    score = decision.get("risk_score", "—")
    default_pct = decision.get("default_probability", "—")
    rec = RECOMMENDATION_LABEL.get(
        str(decision.get("recommendation", "")).upper(),
        str(decision.get("model_decision", "—")),
    )
    st.markdown(
        f"""
        <div class="xai-card">
            <div class="xai-card-header">
                <p class="xai-card-title">Assessment Summary</p>
            </div>
            <p class="xai-assess-title" style="color:{color};">{band} Risk · Score {score}</p>
            <p class="xai-assess-sub">Default probability {default_pct}% · Recommendation: {rec}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inline_bold(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def render_ai_summary_card(summary_md: str) -> None:
    body = _inline_bold(summary_md)
    st.markdown(
        f"""
        <div class="xai-card">
            <div class="xai-card-header">
                <p class="xai-card-title">AI Summary</p>
            </div>
            <p class="xai-summary-body">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_factor_card(title: str, items: list[dict], *, risk: bool) -> None:
    if not items:
        bullets = '<p class="xai-assess-sub" style="margin:0;">No factors identified.</p>'
    else:
        parts = []
        for item in items:
            css = "risk" if risk else "pos"
            icss = "up" if risk else "down"
            sign = "+" if risk else ""
            parts.append(
                f'<div class="xai-bullet {css}"><span>{item["title"]}</span>'
                f'<span class="xai-bullet-impact {icss}">{sign}{item["impact"]:.2f}</span></div>'
            )
        bullets = "".join(parts)

    st.markdown(
        f"""
        <div class="xai-card">
            <div class="xai-card-header">
                <p class="xai-card-title">{title}</p>
            </div>
            {bullets}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mode_selector() -> str:
    _init_explainability_mode()
    has_latest = _has_latest_assessment()

    options = (
        [EXPLAIN_MODE_LATEST, EXPLAIN_MODE_CUSTOM]
        if has_latest
        else [EXPLAIN_MODE_CUSTOM]
    )
    if st.session_state.explainability_mode not in options:
        st.session_state.explainability_mode = options[0]

    return st.radio(
        "Explanation source",
        options=options,
        horizontal=True,
        label_visibility="collapsed",
        key="explainability_mode",
    )


def _resolve_applicant_context(mode: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if mode == EXPLAIN_MODE_LATEST:
        if not _has_latest_assessment():
            return None, None
        return (
            dict(st.session_state["last_applicant_payload"]),
            dict(st.session_state["risk_result"]),
        )

    payload = st.session_state.get("custom_explain_payload")
    if payload:
        return dict(payload), None
    return None, None


def render_custom_applicant_form() -> None:
    from risk_prediction_page import DEFAULT_APPLICANT, _days_employed_from_years, _years_from_days_employed

    d = DEFAULT_APPLICANT
    with st.form("explain_custom_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            ext2 = st.number_input("External credit score 2", 0.0, 1.0, float(d["EXT_SOURCE_2"]), 0.01)
            ext3 = st.number_input("External credit score 3", 0.0, 1.0, float(d["EXT_SOURCE_3"]), 0.01)
            income = st.number_input("Annual income (₹)", 0.0, float(d["AMT_INCOME_TOTAL"]), step=10000.0)
        with c2:
            ext1 = st.number_input("External credit score 1", 0.0, 1.0, float(d["EXT_SOURCE_1"]), 0.01)
            credit = st.number_input("Credit amount (₹)", 0.0, float(d["AMT_CREDIT"]), step=10000.0)
            years = st.number_input(
                "Employment years",
                0.0,
                50.0,
                _years_from_days_employed(int(d["DAYS_EMPLOYED"])),
                0.5,
            )
        submitted = st.form_submit_button("Generate explanation", type="primary", use_container_width=True)

    if submitted:
        st.session_state.custom_explain_payload = {
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": float(d["AMT_ANNUITY"]),
            "AMT_GOODS_PRICE": float(d["AMT_GOODS_PRICE"]),
            "DAYS_BIRTH": int(d["DAYS_BIRTH"]),
            "DAYS_EMPLOYED": _days_employed_from_years(years),
            "REGION_RATING_CLIENT": int(d["REGION_RATING_CLIENT"]),
            "REGION_RATING_CLIENT_W_CITY": int(d["REGION_RATING_CLIENT_W_CITY"]),
            "DAYS_LAST_PHONE_CHANGE": int(d["DAYS_LAST_PHONE_CHANGE"]),
            "DAYS_ID_PUBLISH": int(d["DAYS_ID_PUBLISH"]),
            "REG_CITY_NOT_WORK_CITY": int(d["REG_CITY_NOT_WORK_CITY"]),
            "REG_CITY_NOT_LIVE_CITY": int(d["REG_CITY_NOT_LIVE_CITY"]),
            "FLAG_EMP_PHONE": int(d["FLAG_EMP_PHONE"]),
            "FLAG_DOCUMENT_3": int(d["FLAG_DOCUMENT_3"]),
            "OWN_CAR_AGE": float(d["OWN_CAR_AGE"]),
        }
        st.rerun()


def _estimate_decision_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    model = load_model()
    features = preprocess_applicant(payload)
    prob = float(model.predict_proba(features)[0][1])
    pct = round(prob * 100, 1)
    if prob >= 0.67:
        band = "HIGH"
    elif prob >= 0.40:
        band = "MEDIUM"
    else:
        band = "LOW"
    return {
        "risk_band": band,
        "default_probability": pct,
        "risk_score": int(round(pct)),
        "recommendation": "DECLINE" if band == "HIGH" else "REVIEW" if band == "MEDIUM" else "APPROVE",
    }


def render_individual_column() -> None:
    mode = render_mode_selector()

    if mode == EXPLAIN_MODE_LATEST:
        st.session_state.pop("custom_explain_payload", None)
    else:
        render_custom_applicant_form()

    payload, decision = _resolve_applicant_context(mode)
    if not payload:
        if mode == EXPLAIN_MODE_LATEST:
            st.caption("No assessment available yet.")
        return

    try:
        contributions = compute_contributions(payload)
    except Exception as exc:
        st.error(f"Explanation unavailable: {exc}")
        return

    if decision is None:
        decision = _estimate_decision_from_payload(payload)

    risk_drivers, positive_factors = split_drivers(contributions, payload)
    summary = build_business_summary(
        str(decision.get("risk_band", "MEDIUM")),
        risk_drivers,
        positive_factors,
    )

    render_assessment_summary_card(decision)
    render_ai_summary_card(summary)

    driver_cols = st.columns(2, gap="small")
    with driver_cols[0]:
        render_factor_card("Risk Drivers", risk_drivers, risk=True)
    with driver_cols[1]:
        render_factor_card("Positive Factors", positive_factors, risk=False)


def render_explainability_page(page_header_fn: Callable[..., None]) -> None:
    inject_explainability_styles()

    page_header_fn(
        "Explainability",
        "Portfolio drivers and applicant-level factors behind each credit decision.",
        badge="XAI",
    )

    left_col, right_col = st.columns((2, 3), gap="medium")

    with left_col:
        render_global_feature_importance_card()

    with right_col:
        render_individual_column()
