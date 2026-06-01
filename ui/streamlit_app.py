"""
Credit Risk Intelligence Platform — Streamlit UI shell.

Enterprise dashboard layout with sidebar navigation. Backend integration
is intentionally deferred; pages render placeholders until wired to the API.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------

APP_TITLE = "CreditIQ"
APP_SUBTITLE = "Enterprise Risk Platform"
APP_VERSION = "v0.1.0-shell"

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

NAV_ITEMS: list[tuple[str, str, str]] = [
    ("dashboard", "Dashboard", "📊"),
    ("data_explorer", "Data Explorer", "🗂️"),
    ("risk_prediction", "Risk Prediction", "🎯"),
    ("explainability", "Explainability", "🔍"),
    ("decision_rules", "Decision Rules", "⚖️"),
    ("talk_to_data", "Talk To Data", "💬"),
    ("reports", "Reports", "📑"),
]

PAGE_RENDERERS: dict[str, Callable[[], None]] = {}


def register_page(page_id: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Decorator to register a page render function."""

    def decorator(func: Callable[[], None]) -> Callable[[], None]:
        PAGE_RENDERERS[page_id] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Theme & global styles
# ---------------------------------------------------------------------------


def inject_theme() -> None:
    """Inject dark banking / risk analytics theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap');

        :root {
            --bg-app: #0b0f14;
            --bg-panel: #121820;
            --bg-card: #161d27;
            --bg-card-hover: #1a2330;
            --border: #243044;
            --text: #e8edf5;
            --text-muted: #8b9bb4;
            --accent: #3b82f6;
            --accent-soft: rgba(59, 130, 246, 0.15);
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --radius: 12px;
        }

        .stApp {
            background: linear-gradient(165deg, #0b0f14 0%, #0e1319 45%, #0b1018 100%);
            font-family: 'DM Sans', sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1219 0%, #0a0e14 100%);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        h1, h2, h3, h4, p, label, span {
            font-family: 'DM Sans', sans-serif !important;
        }

        /* Hide default Streamlit chrome for cleaner shell */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* Brand block */
        .brand-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.2;
        }
        .brand-sub {
            font-size: 0.72rem;
            color: var(--text-muted);
            margin: 0.15rem 0 0 0;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .brand-accent {
            color: var(--accent);
        }

        /* Page header */
        .page-header {
            margin-bottom: 1.5rem;
        }
        .page-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text);
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.03em;
        }
        .page-desc {
            font-size: 0.95rem;
            color: var(--text-muted);
            margin: 0;
            max-width: 720px;
            line-height: 1.5;
        }
        .page-badge {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 0.25rem 0.55rem;
            border-radius: 6px;
            background: var(--accent-soft);
            color: var(--accent);
            margin-bottom: 0.5rem;
        }

        /* KPI cards */
        .kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.1rem 1.25rem;
            height: 100%;
            transition: border-color 0.2s, background 0.2s;
        }
        .kpi-card:hover {
            background: var(--bg-card-hover);
            border-color: #2f3f56;
        }
        .kpi-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--text-muted);
            margin: 0 0 0.5rem 0;
        }
        .kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--text);
            margin: 0;
            letter-spacing: -0.02em;
        }
        .kpi-delta {
            font-size: 0.8rem;
            margin: 0.45rem 0 0 0;
            font-weight: 500;
        }
        .kpi-delta.positive { color: var(--success); }
        .kpi-delta.negative { color: var(--danger); }
        .kpi-delta.neutral { color: var(--text-muted); }
        .kpi-icon {
            float: right;
            font-size: 1.25rem;
            opacity: 0.85;
        }

        /* Section panels */
        .panel-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1rem 1.15rem;
            margin-bottom: 0.5rem;
        }
        .panel-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text);
            margin: 0 0 0.25rem 0;
        }
        .panel-subtitle {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin: 0 0 0.75rem 0;
        }

        /* Placeholder banner */
        .placeholder-banner {
            background: rgba(59, 130, 246, 0.08);
            border: 1px dashed #2a4060;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }

        /* Nav hint in sidebar */
        .nav-footer {
            font-size: 0.7rem;
            color: #5c6b82;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }

        /* Streamlit widget tuning */
        .stDataFrame { border-radius: var(--radius); overflow: hidden; }
        div[data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 0.75rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def render_brand_block() -> None:
    """Sidebar brand header."""
    st.markdown(
        f"""
        <p class="brand-title"><span class="brand-accent">Credit</span>IQ</p>
        <p class="brand-sub">{APP_SUBTITLE}</p>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    title: str,
    description: str,
    badge: str | None = None,
) -> None:
    """Consistent page title block."""
    badge_html = f'<span class="page-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="page-header">
            {badge_html}
            <h1 class="page-title">{title}</h1>
            <p class="page-desc">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def placeholder_banner(message: str) -> None:
    """Inform users that a section is not yet wired to the backend."""
    st.markdown(
        f'<div class="placeholder-banner">ℹ️ {message}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    delta: str,
    delta_tone: str = "neutral",
    icon: str = "📈",
) -> None:
    """
    Render a single KPI card.

    delta_tone: 'positive' | 'negative' | 'neutral'
    """
    st.markdown(
        f"""
        <div class="kpi-card">
            <span class="kpi-icon">{icon}</span>
            <p class="kpi-label">{label}</p>
            <p class="kpi-value">{value}</p>
            <p class="kpi-delta {delta_tone}">{delta}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(cards: list[dict[str, str]], columns: int = 4) -> None:
    """Render a responsive row of KPI cards."""
    cols = st.columns(columns)
    for col, card in zip(cols, cards):
        with col:
            kpi_card(**card)


def panel_section(title: str, subtitle: str = "") -> None:
    """Section title inside a content panel."""
    sub = f'<p class="panel-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="panel-card" style="margin-bottom: 0.75rem;">
            <p class="panel-title">{title}</p>
            {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_shell(
    title: str,
    description: str,
    badge: str | None = None,
    show_placeholder: bool = True,
) -> None:
    """Standard wrapper for non-dashboard pages."""
    page_header(title, description, badge=badge)
    if show_placeholder:
        placeholder_banner(
            "UI shell only — backend integration will connect this page to the API."
        )


# ---------------------------------------------------------------------------
# Placeholder data
# ---------------------------------------------------------------------------


def _dashboard_kpis() -> list[dict[str, str]]:
    return [
        {
            "label": "Total Applications",
            "value": "48,291",
            "delta": "▲ +12.4% vs last month",
            "delta_tone": "positive",
            "icon": "📋",
        },
        {
            "label": "Default Rate",
            "value": "8.3%",
            "delta": "▼ 1.1% improvement",
            "delta_tone": "positive",
            "icon": "⚠️",
        },
        {
            "label": "Approval Rate",
            "value": "67.2%",
            "delta": "▲ +2.8% vs last month",
            "delta_tone": "positive",
            "icon": "✅",
        },
        {
            "label": "High Risk Queue",
            "value": "3,842",
            "delta": "▲ +3.1% needs attention",
            "delta_tone": "negative",
            "icon": "🔴",
        },
    ]


def _monthly_applications_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
            "Applications": [8200, 7400, 9100, 6800, 8500],
            "Defaults": [902, 666, 728, 544, 680],
        }
    )


def _risk_band_distribution() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Band": ["Low", "Medium", "High"],
            "Share": [67, 25, 8],
        }
    ).set_index("Band")


def _recent_applications_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Applicant": "Priya Nair",
                "ID": "SK-100291",
                "Loan Amount": "₹3,50,000",
                "Income": "₹72,000/mo",
                "Ext Score": 0.74,
                "Risk Score": 22,
                "Band": "High",
            },
            {
                "Applicant": "Rahul Menon",
                "ID": "SK-100290",
                "Loan Amount": "₹8,00,000",
                "Income": "₹1,40,000/mo",
                "Ext Score": 0.61,
                "Risk Score": 58,
                "Band": "Medium",
            },
            {
                "Applicant": "Anjali Krishnan",
                "ID": "SK-100289",
                "Loan Amount": "₹5,20,000",
                "Income": "₹95,000/mo",
                "Ext Score": 0.82,
                "Risk Score": 81,
                "Band": "Low",
            },
            {
                "Applicant": "Vikram Iyer",
                "ID": "SK-100288",
                "Loan Amount": "₹12,00,000",
                "Income": "₹2,10,000/mo",
                "Ext Score": 0.68,
                "Risk Score": 45,
                "Band": "Medium",
            },
            {
                "Applicant": "Deepa Suresh",
                "ID": "SK-100287",
                "Loan Amount": "₹2,75,000",
                "Income": "₹55,000/mo",
                "Ext Score": 0.79,
                "Risk Score": 74,
                "Band": "Low",
            },
        ]
    )


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------


@register_page("dashboard")
def render_dashboard() -> None:
    page_header(
        "Executive Dashboard",
        "Portfolio overview, default trends, and recent application activity.",
        badge="Overview",
    )

    kpi_row(_dashboard_kpis(), columns=4)

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    chart_left, chart_right = st.columns((3, 2), gap="large")

    with chart_left:
        panel_section(
            "Monthly Applications & Defaults",
            "Placeholder trend — will connect to portfolio analytics.",
        )
        monthly = _monthly_applications_df()
        st.bar_chart(monthly.set_index("Month")[["Applications", "Defaults"]], height=280)

    with chart_right:
        panel_section(
            "Risk Band Distribution",
            "Placeholder mix — will reflect live model scoring.",
        )
        st.bar_chart(_risk_band_distribution(), height=280)
        st.caption("Low 67% · Medium 25% · High 8%")

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    panel_section(
        "Recent Applications",
        "Sample queue — will sync with inference API.",
    )
    table_col, action_col = st.columns((5, 1))
    with table_col:
        st.dataframe(
            _recent_applications_df(),
            use_container_width=True,
            hide_index=True,
            height=220,
        )
    with action_col:
        st.markdown("<div style='height: 2.2rem'></div>", unsafe_allow_html=True)
        st.button("View all", use_container_width=True, disabled=True)
        st.caption("48,291 total")


@register_page("data_explorer")
def render_data_explorer() -> None:
    page_shell(
        "Data Explorer",
        "Browse, filter, and profile the Home Credit application dataset.",
        badge="Data",
    )
    st.markdown("##### Dataset catalog")
    st.info("Dataset tables, column profiles, and filters will appear here.")


@register_page("risk_prediction")
def render_risk_prediction() -> None:
    from risk_prediction_page import render_risk_prediction_page

    render_risk_prediction_page(page_header)


@register_page("explainability")
def render_explainability() -> None:
    from explainability_page import render_explainability_page

    render_explainability_page(page_header)


@register_page("decision_rules")
def render_decision_rules() -> None:
    from decision_rules_page import render_decision_rules_page

    render_decision_rules_page(page_header)


@register_page("talk_to_data")
def render_talk_to_data() -> None:
    page_shell(
        "Talk To Data",
        "Natural-language queries over portfolio and application data.",
        badge="AI",
    )
    st.chat_input("Ask about defaults, regions, or segments…", disabled=True)


@register_page("reports")
def render_reports() -> None:
    page_shell(
        "Reports",
        "Exportable summaries for risk committees and regulatory review.",
        badge="Intelligence",
    )
    st.markdown("##### Report library")
    report_cols = st.columns(3)
    for col, title in zip(
        report_cols,
        ["Monthly Risk Summary", "Default Trend Pack", "Model Performance"],
    ):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption("PDF / CSV export — coming soon")
                st.button("Generate", key=title, disabled=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar & routing
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    """Build sidebar navigation and return selected page id."""
    from app_navigation import apply_pending_navigation

    labels = [f"{icon}  {label}" for _, label, icon in NAV_ITEMS]
    ids = [page_id for page_id, _, _ in NAV_ITEMS]

    # Apply queued navigation before any widget with key="main_nav" is created.
    apply_pending_navigation(ids)

    with st.sidebar:
        render_brand_block()
        st.markdown("<div style='height: 1.25rem'></div>", unsafe_allow_html=True)

        st.markdown(
            '<p style="font-size:0.68rem;font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.08em;color:#5c6b82;margin:0 0 0.5rem 0;">Navigation</p>',
            unsafe_allow_html=True,
        )

        choice = st.radio(
            "Pages",
            options=ids,
            format_func=lambda pid: labels[ids.index(pid)],
            label_visibility="collapsed",
            key="main_nav",
        )

        st.markdown(
            f"""
            <div class="nav-footer">
                {APP_VERSION}<br/>
                LightGBM · ROC-AUC 0.75<br/>
                Shell — no API calls
            </div>
            """,
            unsafe_allow_html=True,
        )

        return choice


def render_top_bar() -> None:
    """Lightweight header row above page content."""
    left, right = st.columns((4, 1))
    with right:
        st.text_input(
            "Search",
            placeholder="Search applicants…",
            label_visibility="collapsed",
            disabled=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} | {APP_SUBTITLE}",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_theme()
    page_id = render_sidebar()
    render_top_bar()

    renderer = PAGE_RENDERERS.get(page_id, render_dashboard)
    renderer()


if __name__ == "__main__":
    main()
