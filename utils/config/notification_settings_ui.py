import streamlit as st

def render_notification_settings(notification_channels, update_callback, test_callback):
    st.subheader("📬 Notification Settings")

    # Dashboard notifications (always available)
    st.checkbox(
        "Show Dashboard Notifications",
        value=notification_channels["dashboard"]["enabled"],
        key="notify_dashboard",
        on_change=lambda: update_callback("dashboard", "enabled", st.session_state["notify_dashboard"])
    )

    # Email notifications
    email_enabled = st.checkbox(
        "Email Notifications",
        value=notification_channels["email"]["enabled"],
        key="notify_email",
        on_change=lambda: update_callback("email", "enabled", st.session_state["notify_email"])
    )

    if email_enabled:
        st.text_input(
            "Email Address",
            value=notification_channels["email"]["address"],
            key="email_address",
            on_change=lambda: update_callback("email", "address", st.session_state["email_address"])
        )

    # SMS notifications
    sms_enabled = st.checkbox(
        "SMS Notifications",
        value=notification_channels["sms"]["enabled"],
        key="notify_sms",
        on_change=lambda: update_callback("sms", "enabled", st.session_state["notify_sms"])
    )

    if sms_enabled:
        st.text_input(
            "Phone Number",
            value=notification_channels["sms"]["number"],
            key="sms_number",
            on_change=lambda: update_callback("sms", "number", st.session_state["sms_number"])
        )

    # Test notifications
    if st.button("Test Notifications"):
        test_callback()