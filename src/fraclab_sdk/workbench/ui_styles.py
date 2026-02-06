"""Shared UI styling utility for Streamlit pages."""

import streamlit as st


def apply_global_styles():
    """Inject global CSS to hide Streamlit UI elements and apply common styling."""
    st.markdown("""
    <style>
        /* =================================
           1. Hide Streamlit UI Elements
           ================================= */

        /* Hide top header bar */
        header[data-testid="stHeader"] {
            display: none !important;
        }

        /* Hide Deploy button; keep toolbar for sidebar toggle */
        .stDeployButton {
            display: none !important;
        }

        /* Force sidebar visible and disable collapse control */
        [data-testid="stSidebar"] {
            transform: none !important;
            min-width: 260px;
        }
        [data-testid="stSidebarNav"] {
            min-width: 240px;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        /* Hide footer (Made with Streamlit) */
        footer {
            display: none !important;
        }

        /* Hide element toolbar on tables/dataframes */
        [data-testid="stElementToolbar"] {
            display: none !important;
        }

        /* Adjust main content padding (header hidden, content needs padding) */
        .main .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }

        /* =================================
           2. Button Styling
           ================================= */

        div[data-testid="stButton"] button {
            white-space: nowrap !important;
            border-radius: 6px !important;
            min-width: 60px !important;
            font-weight: 500 !important;
        }

        /* =================================
           3. Anti-Copy Protection
           ================================= */

        /* Disable text selection on key areas */
        .element-container,
        [data-testid="stDataFrame"],
        [data-testid="stDataEditor"],
        [data-testid="stCode"],
        [data-testid="stJson"] {
            user-select: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
        }

        /* Hide copy buttons on st.code / st.json */
        [data-testid="stCode"] button,
        [data-testid="stJson"] button {
            display: none !important;
        }

        /* =================================
           4. Component Styling
           ================================= */

        /* Expander header styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            background-color: #f9f9fb;
            border-radius: 6px;
        }

        /* Data Editor border styling */
        [data-testid="stDataEditor"] {
            border: 1px solid #e6e9ef;
            border-radius: 6px;
        }

    </style>
    """, unsafe_allow_html=True)
