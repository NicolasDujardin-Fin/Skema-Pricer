"""
Skema Pricer — Streamlit entry point.
Run with: streamlit run app.py
"""

import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import streamlit as st

from ui.components.shared import inject_css

_LOGO_PATH = os.path.join(_ROOT, "assets", "skema_logo.png")
from ui.tabs.options import options_tab
from ui.tabs.bonds import bonds_tab
from ui.tabs.turbo import turbo_tab
from ui.tabs.discount import discount_tab
from ui.tabs.bonus import bonus_tab
from ui.tabs.interview import interview_tab

# ── Page config & CSS ──
st.set_page_config(page_title="Skema Pricer", layout="wide", page_icon="📈",
                    initial_sidebar_state="expanded")
inject_css()

# ── Logo ──
if os.path.exists(_LOGO_PATH):
    st.logo(_LOGO_PATH)

# ── Navigation ──
with st.sidebar:
    st.markdown(
        '<span class="app-title">Skema Pricer</span>'
        '<span class="app-badge">Derivatives Pricer</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    active_tab = st.radio(
        "Navigation",
        ["Options", "Bonds", "Turbo", "Discount Cert.", "Bonus Cert.", "Interview"],
        horizontal=True, key="nav", label_visibility="collapsed",
    )
    st.markdown("---")

# ── Render active tab ──
if active_tab == "Options":
    options_tab()
elif active_tab == "Bonds":
    bonds_tab()
elif active_tab == "Turbo":
    turbo_tab()
elif active_tab == "Discount Cert.":
    discount_tab()
elif active_tab == "Bonus Cert.":
    bonus_tab()
else:
    interview_tab()
