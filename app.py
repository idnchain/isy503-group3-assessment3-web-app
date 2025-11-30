import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
import os
import gdown

# --- CONFIGURATION ---
GOOGLE_DRIVE_FILE_ID = '1RBW03KAggfsgw8JAIYc98pSzadAkDPFn' 
MODEL_FILENAME = 'best_bert_model.keras'
MODEL_NAME = 'bert-base-uncased'

# --- PAGE SETUP ---
st.set_page_config(
    page_title="BERT Sentiment Analyser",
    page_icon=":streamlit:",
    layout="centered"
)
st.title("Multi-domain Sentiment Dataset Analyser")
st.markdown("Enter a sentence to analyse if the sentiment is **Positive** or **Negative**.")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_model_and_tokenizer():
    # A. Download Model from Drive if it doesn't exist locally
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading model weights from Google Drive... (This happens only once)"):
            try:
                url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
                gdown.download(url, MODEL_FILENAME, quiet=False)
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                st.stop()
            
    # B. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        st.stop()
    
    # C. Instantiate Architecture
    try:
        model = TFAutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            from_pt=False, 
            use_safetensors=False
        )
        
        # D. Load the weights
        model.load_weights(MODEL_FILENAME)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()
        
    return model, tokenizer

# Load the model
model, tokenizer = load_model_and_tokenizer()

# --- PREDICTION FUNCTION ---
def predict_sentiment_bert(sentence, model, tokenizer):
    MAX_BERT_SEQUENCE_LEN = 128
    
    # Tokenize
    encoded_input = tokenizer.batch_encode_plus(
        [sentence],
        add_special_tokens=True,
        max_length=MAX_BERT_SEQUENCE_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='tf'
    )

    # Predict
    logits = model(encoded_input).logits
    
    # Calculate probabilities (optional, for confidence score)
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
    
    # Get label index
    prediction_index = tf.argmax(logits, axis=1).numpy()[0]

    if prediction_index == 1:
        return "Positive", probabilities[1]
    else:
        return "Negative", probabilities[0]

# --- USER INTERFACE ---
user_input = st.text_area("Customer Review:", height=150, placeholder="Example: The delivery was fast but the product quality is poor.")
analyze_btn = st.button("Analyze Sentiment", type="primary")

if analyze_btn and user_input:
    label, confidence = predict_sentiment_bert(user_input, model, tokenizer)
    
    st.divider()
    if label == "Positive":
        st.success(f"### Prediction: {label} Review")
        st.metric("Confidence Score", f"{confidence:.2%}")
    else:
        st.error(f"### Prediction: {label} Review")
        st.metric("Confidence Score", f"{confidence:.2%}")

elif analyze_btn and not user_input:
    st.warning("Please enter some text to analyze.")