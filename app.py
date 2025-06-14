import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy

# Import your preprocessing functions
from preprocessing import text_strip, spacy_cleaning

import tensorflow as tf



# Set up default graph globally
graph = tf.get_default_graph()

@st.cache(allow_output_mutation=True)
def load_resources():
    global graph
    with graph.as_default():
        encoder = load_model("encoder_model.h5")
        decoder = load_model("decoder_model.h5")

        with open("x_tokenizer.pkl", "rb") as f:
            x_tokenizer = pickle.load(f)

        with open("y_tokenizer.pkl", "rb") as f:
            y_tokenizer = pickle.load(f)

        with open("preprocessing_params.pkl", "rb") as f:
            params = pickle.load(f)

    return encoder, decoder, x_tokenizer, y_tokenizer, params

# Load once
encoder_model, decoder_model, x_tokenizer, y_tokenizer, params = load_resources()

max_text_len = params["max_text_len"]
max_summary_len = params["max_summary_len"]
target_word_index = y_tokenizer.word_index
reverse_target_word_index = y_tokenizer.index_word

# --- INPUT CLEANING FUNCTIONS ---
def preprocess_input(text):
    cleaned_text_gen = text_strip([text])
    cleaned_text_list = spacy_cleaning(cleaned_text_gen, is_summary=False)
    cleaned_text = cleaned_text_list[0]
    seq = x_tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_text_len, padding='post')
    return padded

# --- SEQ2SEQ PREDICTION ---
def decode_sequence(input_seq):
    with graph.as_default():  # <== Critical
        e_out, e_h, e_c = encoder_model.predict(input_seq)
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = target_word_index['sostok']

        decoded_sentence = ''
        stop_condition = False
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + [e_h, e_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index.get(sampled_token_index, '')

            if sampled_token != 'eostok' and sampled_token != '':
                decoded_sentence += ' ' + sampled_token

            if sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1):
                stop_condition = True

            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index
            e_h, e_c = h, c

    return decoded_sentence.strip()

# Streamlit app for news article summarization
import streamlit as st

import streamlit as st

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
    }
    .stTextArea textarea {
        font-size: 16px;
        line-height: 1.6;
    }
    .summary-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }
    .centered-button {
        display: flex;
        justify-content: center;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>📰 News Article Summarizer</h1>", unsafe_allow_html=True)
st.markdown("#### ✨ Enter a news article below and get a concise summary instantly.")

# Input area
st.markdown("### 📝 Your Article")
# Input Area
article = st.text_area("📝 Paste your article below:", height=250)

# Generate Button (centered)
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
generate = st.button("✨ Generate Summary")
st.markdown("</div>", unsafe_allow_html=True)

# Summary Output
if generate:
    if article.strip() == "":
        st.warning("⚠️ Please enter some text to summarize!")
    else:
        with st.spinner("🤖 Generating summary..."):
            input_seq = preprocess_input(article)
            summary = decode_sequence(input_seq)

        st.markdown("### ✅ Summary")
        st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
        st.success("Summary generated successfully!")
