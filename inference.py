import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import text_strip, spacy_cleaning

# Load encoder and decoder inference models
encoder_model = load_model("encoder_model.h5")
decoder_model = load_model("decoder_model.h5")

# Load tokenizers
with open('x_tokenizer.pkl', 'rb') as f:
    x_tokenizer = pickle.load(f)
with open('y_tokenizer.pkl', 'rb') as f:
    y_tokenizer = pickle.load(f)

# Load preprocessing params (e.g. max lengths)
with open("preprocessing_params.pkl", "rb") as f:
    params = pickle.load(f)

max_text_len = params["max_text_len"]
max_summary_len = params["max_summary_len"]

# Convert raw text input to padded sequence
def preprocess_input(text):
    # Clean input text using preprocessing.py functions
    cleaned_text_gen = text_strip([text])
    cleaned_text = spacy_cleaning(cleaned_text_gen, is_summary=False)  # returns list
    
    # Tokenize cleaned text and pad
    seq = x_tokenizer.texts_to_sequences(cleaned_text)
    padded = pad_sequences(seq, maxlen=max_text_len, padding='post')
    return padded

# Vocab dictionaries
target_word_index = y_tokenizer.word_index
reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word


# Decode predicted sequence
def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = target_word_index['sostok']

    decoded_sentence = ''
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

        if sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1):
            stop_condition = True

        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        e_h, e_c = h, c

    return decoded_sentence.strip()

# Util
