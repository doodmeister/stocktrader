import streamlit as st
import pandas as pd
from ml_pipeline import train_model  # Assuming you have this function
from model_manager import save_model  # Assuming you have this

st.title("🧠 Model Training and Deployment")

uploaded_file = st.file_uploader("Upload Training Data (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data.head())

    st.subheader("Training Configuration")
    epochs = st.slider("Epochs", 1, 100, 10)
    learning_rate = st.number_input("Learning Rate", value=0.001, format="%.5f")

    if st.button("Train Model"):
        st.info("Training model...")
        model, metrics = train_model(data, epochs=epochs, learning_rate=learning_rate)

        st.success("Training completed!")
        st.json(metrics)

        if st.button("Save Trained Model"):
            save_model(model, "models/pattern_nn_v1.pth")
            st.success("Model saved successfully!")
