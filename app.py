import streamlit as st
import joblib
import pandas as pd
from utils import NepaliTextProcessor  # assuming your class is in this file
import os

# ------------------------------
# Load vectorizer and models
# ------------------------------
MODEL_PATH = r"C:\Users\Nisha\Desktop\telecom_sentiment_inference_code\models"

VECTORIZER_PATH = os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl")

# Load vectorizer
vectorizer = joblib.load(VECTORIZER_PATH)

# Load available models dynamically
available_models = {}
for fname in os.listdir(MODEL_PATH):
    if fname.endswith(".pkl") and fname != "tfidf_vectorizer.pkl":
        model_name = fname.replace(".pkl", "").replace("_", " ")
        available_models[model_name] = joblib.load(os.path.join(MODEL_PATH, fname))

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📊 Nepali/English Sentiment Classifier")
st.write("Choose a model and enter your text to analyze sentiment.")

# Sidebar - model choice
model_choice = st.sidebar.selectbox("Select Model", list(available_models.keys()))

# Text input
user_input = st.text_area("✍️ Enter your text:", height=150)

# Define label mapping
label_map = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

# Process input and predict
if st.button("🔮 Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        processor = NepaliTextProcessor()
        row = pd.Series({"Commnets": user_input, "Sentiment": ""})
        processed = processor.process_row(row)

        cleaned_text = processed["cleaned_text"]

        # Vectorize
        X_input = vectorizer.transform([cleaned_text])

        # Get model
        model = available_models[model_choice]

        # Predict
        y_pred = model.predict(X_input)[0]

        # Try probabilities if model supports it
        proba_text = ""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[0]
            proba_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probs
            })
            st.subheader("Prediction Confidence")
            st.dataframe(proba_df)

        # Show results
        st.subheader("🔎 Analysis Result")
        st.write(f"**Cleaned Text:** {cleaned_text}")
        # st.write(f"**Predicted Sentiment ({model_choice}):** {y_pred}")

        # Convert prediction to readable text
        sentiment_label = label_map.get(int(y_pred), "Unknown")
        st.write(f"**Predicted Sentiment ({model_choice}):** {sentiment_label}")

        # Extra: show extracted emojis + sentiment
        st.write(f"**Detected Emojis:** {processed['emoticons']}")
        st.write(f"**Emoji Sentiment:** {processed['emoticon_sentiment']} ({processed['emoticon_score']})")
        st.write(f"**Text Sentiment (rule-based/TextBlob):** {processed['text_sentiment']} ({processed['text_score']})")
        st.write(f"**Combined Sentiment:** {processed['combined_sentiment']} ({processed['combined_score']})")
        # st.write(f"**Model Used:** {model_choice}")