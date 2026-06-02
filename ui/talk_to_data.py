"""
AI Data Analyst — deterministic NL → SQL over Home Credit application_train.

No external LLMs. Uses pandas, sqlite3, and keyword intent matching.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.portfolio_loader import resolve_portfolio_csv  # noqa: E402

SNAPSHOT_PATH = PROJECT_ROOT / "models" / "portfolio_scoring_snapshot.json"
TABLE_NAME = "application_train"

SUGGESTED_QUESTIONS: list[tuple[str, str]] = [
    ("Approval Rate", "What is the approval rate?"),
    ("Default Rate", "What is the default rate?"),
    ("Portfolio Summary", "Summarize portfolio"),
    ("Average Income", "What is the average income?"),
    ("Loan Type Risk", "Which loan type has highest default rate?"),
    ("Risk By Gender", "Show risk by gender"),
    ("Top Risk Regions", "What are the top 5 highest risk regions?"),
]


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    sql: str
    title: str


@dataclass
class QueryResult:
    plan: QueryPlan
    frame: pd.DataFrame
    insight: str
    error: str | None = None


def _normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


@st.cache_data(show_spinner="Loading Home Credit application data…")
def load_data() -> pd.DataFrame:
    path = resolve_portfolio_csv()
    if path is None:
        raise FileNotFoundError(
            "application_train.csv not found. Place it at data/application_train.csv "
            "or set CREDIT_RISK_PORTFOLIO_CSV."
        )
    return pd.read_csv(path)


def _load_portfolio_kpis() -> dict[str, float]:
    """Cached underwriting KPIs from portfolio batch scoring (display / SQL seed)."""
    if not SNAPSHOT_PATH.is_file():
        return {}
    with SNAPSHOT_PATH.open(encoding="utf-8") as file:
        snap = json.load(file)
    return {
        "approval_rate_pct": float(snap.get("approval_rate_pct", 0)),
        "decline_rate_pct": float(snap.get("decline_rate_pct", 0)),
        "review_rate_pct": float(snap.get("review_rate_pct", 0)),
        "observed_default_rate_pct": float(snap.get("observed_default_rate_pct", 0)),
        "scored_records": float(snap.get("scored_records", 0)),
        "threshold": float(snap.get("threshold", 0.67)),
    }


@st.cache_resource(show_spinner="Preparing in-memory analytics database…")
def create_database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    frame = load_data()
    frame.to_sql(TABLE_NAME, conn, index=False, if_exists="replace")

    kpis = _load_portfolio_kpis()
    conn.execute(
        """
        CREATE TABLE portfolio_kpis (
            metric TEXT PRIMARY KEY,
            value REAL NOT NULL
        )
        """
    )
    if kpis:
        rows = [
            ("approval_rate_pct", kpis["approval_rate_pct"]),
            ("decline_rate_pct", kpis["decline_rate_pct"]),
            ("review_rate_pct", kpis["review_rate_pct"]),
            ("observed_default_rate_pct", kpis["observed_default_rate_pct"]),
            ("scored_records", kpis["scored_records"]),
            ("decision_threshold", kpis["threshold"]),
        ]
        conn.executemany("INSERT INTO portfolio_kpis (metric, value) VALUES (?, ?)", rows)
    conn.commit()
    return conn


def generate_sql(question: str) -> QueryPlan:
    """Map natural language to deterministic SQL via keyword intents."""
    q = _normalize(question)

    if not q:
        return QueryPlan(
            "unknown",
            "-- Ask a supported portfolio question using the chips or chat box.",
            "Unknown",
        )

    if any(k in q for k in ("summarize portfolio", "portfolio summary", "summarize the portfolio")):
        return QueryPlan(
            "portfolio_summary",
            f"""-- Portfolio summary (multiple metrics)
SELECT COUNT(*) AS total_applications FROM {TABLE_NAME};

SELECT AVG(TARGET) AS default_rate FROM {TABLE_NAME};

SELECT AVG(AMT_INCOME_TOTAL) AS average_income FROM {TABLE_NAME};

SELECT NAME_CONTRACT_TYPE,
       COUNT(*) AS applications,
       ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct
FROM {TABLE_NAME}
GROUP BY NAME_CONTRACT_TYPE
ORDER BY applications DESC;""",
            "Portfolio Summary",
        )

    if any(k in q for k in ("approval rate", "approval %", "approved", "approve rate")):
        threshold = _load_portfolio_kpis().get("threshold", 0.67)
        return QueryPlan(
            "approval_rate",
            f"""-- Approval rate from batch LightGBM scoring (recommendation = APPROVE)
-- Decision threshold: {threshold:.2f}
SELECT value AS approval_rate_pct
FROM portfolio_kpis
WHERE metric = 'approval_rate_pct';""",
            "Approval Rate",
        )

    if any(
        k in q
        for k in (
            "loan type",
            "contract type",
            "highest default",
            "default by loan",
            "which loan",
        )
    ):
        return QueryPlan(
            "loan_type_default",
            f"""SELECT NAME_CONTRACT_TYPE,
       ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct,
       COUNT(*) AS applications
FROM {TABLE_NAME}
GROUP BY NAME_CONTRACT_TYPE
ORDER BY default_rate_pct DESC;""",
            "Loan Type Default Rate",
        )

    if any(k in q for k in ("default rate", "default risk", "default %")) and "loan" not in q:
        return QueryPlan(
            "default_rate",
            f"""SELECT ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct
FROM {TABLE_NAME};""",
            "Default Rate",
        )

    if any(
        k in q
        for k in (
            "how many application",
            "number of application",
            "total application",
            "count application",
            "how many applicants",
            "how many loans",
        )
    ):
        return QueryPlan(
            "application_count",
            f"""SELECT COUNT(*) AS total_applications
FROM {TABLE_NAME};""",
            "Application Count",
        )

    if any(k in q for k in ("average income", "mean income", "avg income", "income average")):
        return QueryPlan(
            "average_income",
            f"""SELECT ROUND(AVG(AMT_INCOME_TOTAL), 2) AS average_income
FROM {TABLE_NAME};""",
            "Average Income",
        )

    if any(
        k in q
        for k in (
            "under 30",
            "below 30",
            "younger than 30",
            "less than 30",
            "applicants under 30",
        )
    ):
        return QueryPlan(
            "under_30",
            f"""SELECT COUNT(*) AS applicants_under_30
FROM {TABLE_NAME}
WHERE ABS(DAYS_BIRTH) / 365.25 < 30;""",
            "Applicants Under 30",
        )

    if any(k in q for k in ("gender", "by gender", "male", "female", "risk by gender")):
        return QueryPlan(
            "risk_by_gender",
            f"""SELECT CODE_GENDER,
       COUNT(*) AS applications,
       ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct
FROM {TABLE_NAME}
GROUP BY CODE_GENDER
ORDER BY default_rate_pct DESC;""",
            "Risk by Gender",
        )

    if any(
        k in q
        for k in (
            "top 5",
            "top five",
            "highest risk region",
            "risk region",
            "risk by region",
            "regional risk",
        )
    ) and "gender" not in q:
        return QueryPlan(
            "top_risk_regions",
            f"""SELECT
    'Rating ' || CAST(REGION_RATING_CLIENT AS TEXT)
        || ' / City ' || CAST(REGION_RATING_CLIENT_W_CITY AS TEXT) AS region,
    COUNT(*) AS applications,
    ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct
FROM {TABLE_NAME}
GROUP BY REGION_RATING_CLIENT, REGION_RATING_CLIENT_W_CITY
HAVING COUNT(*) >= 5000
ORDER BY default_rate_pct DESC
LIMIT 5;""",
            "Top 5 Highest Risk Regions",
        )

    return QueryPlan(
        "unknown",
        f"-- No matching intent for: {question}\n"
        "-- Try: default rate, approval rate, average income, loan type risk, summarize portfolio.",
        "Unsupported Question",
    )


def execute_query(conn: sqlite3.Connection, plan: QueryPlan) -> pd.DataFrame:
    """Run SQL for the matched intent."""
    if plan.intent == "unknown":
        return pd.DataFrame()

    if plan.intent == "portfolio_summary":
        total = int(
            pd.read_sql_query(f"SELECT COUNT(*) AS v FROM {TABLE_NAME}", conn).iloc[0, 0]
        )
        default_pct = float(
            pd.read_sql_query(
                f"SELECT ROUND(100.0 * AVG(TARGET), 2) AS v FROM {TABLE_NAME}", conn
            ).iloc[0, 0]
        )
        avg_income = float(
            pd.read_sql_query(
                f"SELECT ROUND(AVG(AMT_INCOME_TOTAL), 2) AS v FROM {TABLE_NAME}", conn
            ).iloc[0, 0]
        )
        loans = pd.read_sql_query(
            f"""
            SELECT NAME_CONTRACT_TYPE AS loan_type,
                   COUNT(*) AS applications,
                   ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct
            FROM {TABLE_NAME}
            GROUP BY NAME_CONTRACT_TYPE
            ORDER BY applications DESC
            """,
            conn,
        )
        overview = pd.DataFrame(
            [
                {"section": "Portfolio KPIs", "metric": "Total Applications", "value": total},
                {"section": "Portfolio KPIs", "metric": "Default Rate (%)", "value": default_pct},
                {"section": "Portfolio KPIs", "metric": "Average Income", "value": avg_income},
            ]
        )
        overview["detail"] = ""
        loan_rows = pd.DataFrame(
            {
                "section": "Loan Type Distribution",
                "metric": loans["loan_type"],
                "value": loans["applications"],
                "detail": loans["default_rate_pct"].astype(str) + "% default rate",
            }
        )
        return pd.concat([overview, loan_rows], ignore_index=True)

    cleaned = "\n".join(
        line for line in plan.sql.splitlines() if not line.strip().startswith("--")
    ).strip()
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    if not statements:
        return pd.DataFrame()
    return pd.read_sql_query(statements[-1], conn)


def generate_insight(plan: QueryPlan, frame: pd.DataFrame) -> str:
    """Plain-English interpretation of query results."""
    if plan.intent == "unknown" or frame.empty:
        return (
            "I can answer a focused set of portfolio questions. "
            "Use the suggested chips or ask about default rate, approval rate, income, loan types, or gender."
        )

    if plan.intent == "approval_rate":
        val = float(frame.iloc[0, 0])
        return (
            f"The portfolio approval rate is **{val:.1f}%** under the current LightGBM underwriting policy "
            f"(batch-scored recommendations). Applicants classified as APPROVE are cleared for origination; "
            "MEDIUM-risk cases are routed to manual review."
        )

    if plan.intent == "default_rate":
        val = float(frame.iloc[0, 0])
        return (
            f"The portfolio default rate is **{val:.1f}%**, indicating a relatively low but meaningful "
            "credit risk exposure in the training portfolio (TARGET = 1)."
        )

    if plan.intent == "application_count":
        val = int(frame.iloc[0, 0])
        return f"The dataset contains **{val:,}** credit applications in `application_train`."

    if plan.intent == "loan_type_default":
        top = frame.iloc[0]
        loan = str(top["NAME_CONTRACT_TYPE"])
        rate = float(top["default_rate_pct"])
        return (
            f"**{loan}** shows the highest default rate at **{rate:.1f}%**. "
            "Cash loans typically warrant tighter underwriting and monitoring relative to other products."
        )

    if plan.intent == "average_income":
        val = float(frame.iloc[0, 0])
        return (
            f"Average stated income is **₹{val:,.0f}**, useful for affordability and "
            "annuity-to-income policy checks during underwriting."
        )

    if plan.intent == "under_30":
        val = int(frame.iloc[0, 0])
        return (
            f"**{val:,}** applicants are under 30 years old (derived from `DAYS_BIRTH`). "
            "Younger borrowers may show different employment stability and default patterns."
        )

    if plan.intent == "risk_by_gender":
        parts = []
        for _, row in frame.iterrows():
            parts.append(
                f"{row['CODE_GENDER']}: {float(row['default_rate_pct']):.1f}% default "
                f"({int(row['applications']):,} applications)"
            )
        return "Default rates by gender — " + "; ".join(parts) + "."

    if plan.intent == "top_risk_regions":
        parts = []
        for _, row in frame.iterrows():
            parts.append(
                f"{row['region']}: {float(row['default_rate_pct']):.1f}% default "
                f"({int(row['applications']):,} applications)"
            )
        return "Highest-risk regions (client / city rating) — " + "; ".join(parts) + "."

    if plan.intent == "portfolio_summary":
        kpi = frame[frame["section"] == "Portfolio KPIs"]
        total = int(kpi.loc[kpi["metric"] == "Total Applications", "value"].iloc[0])
        dr = float(kpi.loc[kpi["metric"] == "Default Rate (%)", "value"].iloc[0])
        inc = float(kpi.loc[kpi["metric"] == "Average Income", "value"].iloc[0])
        return (
            f"Portfolio overview: **{total:,}** applications, **{dr:.1f}%** observed default rate, "
            f"and **₹{inc:,.0f}** average income. Loan-type rows in the results table show product mix "
            "and default rates for underwriting segmentation."
        )

    return "Query executed successfully. Review the table for detailed values."


def run_analyst_query(question: str) -> QueryResult:
    plan = generate_sql(question)
    try:
        conn = create_database()
        frame = execute_query(conn, plan)
        insight = generate_insight(plan, frame)
        return QueryResult(plan=plan, frame=frame, insight=insight)
    except Exception as exc:
        return QueryResult(
            plan=plan,
            frame=pd.DataFrame(),
            insight="Unable to complete the query against the in-memory database.",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def inject_talk_to_data_styles() -> None:
    st.markdown(
        """
        <style>
        .tda-hero {
            margin-bottom: 1rem; padding-bottom: 0.85rem;
            border-bottom: 1px solid #243044;
        }
        .tda-eyebrow {
            font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.12em; color: #3b82f6; margin: 0 0 0.35rem 0;
        }
        .tda-title {
            font-size: 1.75rem; font-weight: 700; color: #f1f5f9; margin: 0;
            letter-spacing: -0.03em;
        }
        .tda-sub {
            font-size: 0.88rem; color: #8b9bb4; margin: 0.4rem 0 0 0; line-height: 1.5;
        }
        .tda-panel-label {
            font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.08em; color: #6b7c96; margin: 0 0 0.5rem 0;
        }
        .tda-bubble-user {
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.35);
            border-radius: 10px 10px 10px 2px;
            padding: 0.65rem 0.85rem; margin: 0.5rem 0;
            font-size: 0.84rem; color: #e8edf5;
        }
        .tda-bubble-assistant {
            background: #161d27;
            border: 1px solid #243044;
            border-radius: 10px 10px 2px 10px;
            padding: 0.65rem 0.85rem; margin: 0.5rem 0 0.75rem 0;
            font-size: 0.82rem; color: #c5d0e0; line-height: 1.5;
        }
        .tda-insight {
            background: linear-gradient(135deg, #121820 0%, #161d27 100%);
            border: 1px solid #2a4060;
            border-left: 3px solid #3b82f6;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            font-size: 0.82rem; color: #d1dae8; line-height: 1.55;
            margin-top: 0.5rem;
        }
        .tda-chip-row { margin: 0.35rem 0 0.75rem 0; }
        div[data-testid="stHorizontalBlock"] .stButton > button {
            font-size: 0.72rem !important;
            padding: 0.25rem 0.55rem !important;
            border-radius: 999px !important;
            background: #121820 !important;
            border: 1px solid #243044 !important;
            color: #94a3b8 !important;
        }
        div[data-testid="stHorizontalBlock"] .stButton > button:hover {
            border-color: #3b82f6 !important;
            color: #e8edf5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_chat_state() -> None:
    if "tda_messages" not in st.session_state:
        st.session_state.tda_messages = []
    if "tda_active_result" not in st.session_state:
        st.session_state.tda_active_result = None


def _process_question(question: str) -> None:
    question = question.strip()
    if not question:
        return
    st.session_state.tda_messages.append({"role": "user", "content": question})
    result = run_analyst_query(question)
    st.session_state.tda_active_result = result
    st.session_state.tda_messages.append(
        {
            "role": "assistant",
            "content": result.insight,
            "sql": result.plan.sql,
            "title": result.plan.title,
        }
    )


def _render_chat_history() -> None:
    for msg in st.session_state.tda_messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="tda-bubble-user"><strong>You</strong><br/>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.container(border=True):
                st.caption("Analyst")
                st.markdown(msg["content"])


def _render_suggested_chips() -> None:
    st.markdown('<p class="tda-panel-label">Suggested questions</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (label, question) in enumerate(SUGGESTED_QUESTIONS):
        with cols[idx % 3]:
            if st.button(label, key=f"tda_chip_{idx}", use_container_width=True):
                _process_question(question)
                st.rerun()


def _render_analytics_panel(result: QueryResult | None) -> None:
    st.markdown('<p class="tda-panel-label">Query workspace</p>', unsafe_allow_html=True)

    if result is None:
        st.info("Ask a question or select a chip to generate SQL and portfolio analytics.")
        return

    if result.error:
        st.error(result.error)

    st.markdown("**Generated SQL**")
    st.code(result.plan.sql, language="sql")

    st.markdown("**Results**")
    if result.frame.empty:
        st.caption("No rows returned.")
    else:
        st.dataframe(result.frame, use_container_width=True, hide_index=True, height=min(320, 48 + 35 * len(result.frame)))

    st.markdown('<p class="tda-panel-label" style="margin-top:0.75rem">Business insight</p>', unsafe_allow_html=True)
    st.markdown(result.insight)


def render_talk_to_data_page(page_header_fn: Callable[..., None]) -> None:
    del page_header_fn
    inject_talk_to_data_styles()
    _init_chat_state()

    st.markdown(
        """
        <div class="tda-hero">
            <p class="tda-eyebrow">Enterprise Risk Platform</p>
            <h1 class="tda-title">AI Data Analyst</h1>
            <p class="tda-sub">
                Ask questions about portfolio performance, risk distribution and applicant characteristics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        create_database()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Failed to initialize analytics database: {exc}")
        return

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<p class="tda-panel-label">Conversation</p>', unsafe_allow_html=True)
        _render_suggested_chips()

        chat_box = st.container(height=380, border=False)
        with chat_box:
            if not st.session_state.tda_messages:
                st.caption("Start with a suggested question or type your own below.")
            else:
                _render_chat_history()

        prompt = st.chat_input("Ask about defaults, approval rate, income, loan types…")
        if prompt:
            _process_question(prompt)
            st.rerun()

        if st.button("Clear conversation", type="secondary"):
            st.session_state.tda_messages = []
            st.session_state.tda_active_result = None
            st.rerun()

    with right:
        _render_analytics_panel(st.session_state.tda_active_result)

    st.caption(
        "Deterministic NL→SQL engine · in-memory SQLite · `application_train` · no external LLM APIs"
    )
