import streamlit as st

def initialize_dashboard_session_state():
    """Initialize common Streamlit session state variables if they don't exist."""
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = None
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = None
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    if 'login_status' not in st.session_state:
        st.session_state.login_status = False