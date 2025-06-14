from tensorflow.keras.models import load_model

# Load models (original ones you trained or got from someone)
encoder = load_model("encoder_model.h5", compile=False)
decoder = load_model("decoder_model.h5", compile=False)

# Re-save them in a legacy-compatible format
encoder.save("encoder_model_legacy.h5")
decoder.save("decoder_model_legacy.h5")

print("✅ Models successfully re-saved.")