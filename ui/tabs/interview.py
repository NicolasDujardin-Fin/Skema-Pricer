"""Interview Q&A tab — standalone questions for interview prep."""

import streamlit as st

from ui.components.shared import section, render_qa


def interview_tab():
    with st.sidebar:
        st.markdown('<p class="app-title">Interview Q&A</p>', unsafe_allow_html=True)

    render_qa("interview_greeks")
    st.markdown("")
    render_qa("interview_it")
