import streamlit as st
import os

st.title("⚙️ Settings and Configuration")

st.subheader("API Keys")
etrade_consumer_key = st.text_input("E*TRADE Consumer Key", value=os.getenv("ETRADE_CONSUMER_KEY", ""))
etrade_consumer_secret = st.text_input("E*TRADE Consumer Secret", value=os.getenv("ETRADE_CONSUMER_SECRET", ""))
sandbox_mode = st.checkbox("Use Sandbox Mode", value=os.getenv("ETRADE_SANDBOX", "True") == "True")

st.subheader("Email Notifications")
smtp_server = st.text_input("SMTP Server", value=os.getenv("SMTP_SERVER", "smtp.example.com"))
smtp_port = st.number_input("SMTP Port", value=int(os.getenv("SMTP_PORT", 587)))
smtp_username = st.text_input("SMTP Username", value=os.getenv("SMTP_USERNAME", ""))
smtp_password = st.text_input("SMTP Password", type="password")
from_email = st.text_input("From Email", value=os.getenv("FROM_EMAIL", ""))
to_email = st.text_input("To Email", value=os.getenv("TO_EMAIL", ""))

st.subheader("Risk Management")
max_position_size = st.number_input("Max Position Size ($)", min_value=100, value=5000)
stop_loss_pct = st.slider("Stop Loss %", min_value=1, max_value=50, value=10)
take_profit_pct = st.slider("Take Profit %", min_value=1, max_value=50, value=20)

if st.button("Save Settings"):
    with open(".env", "w") as f:
        f.write(f"ETRADE_CONSUMER_KEY={etrade_consumer_key}\n")
        f.write(f"ETRADE_CONSUMER_SECRET={etrade_consumer_secret}\n")
        f.write(f"ETRADE_SANDBOX={sandbox_mode}\n")
        f.write(f"SMTP_SERVER={smtp_server}\n")
        f.write(f"SMTP_PORT={smtp_port}\n")
        f.write(f"SMTP_USERNAME={smtp_username}\n")
        f.write(f"SMTP_PASSWORD={smtp_password}\n")
        f.write(f"FROM_EMAIL={from_email}\n")
        f.write(f"TO_EMAIL={to_email}\n")
    st.success("Settings saved to `.env` file!")
