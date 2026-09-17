import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import gradio as gr

IMG_SIZE = 160
LATENT_DIM = 64
HV_DIM = 10000

# 1. Load Models & Prototypes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encoder = tf.keras.models.load_model(os.path.join(BASE_DIR, "encoder_model.keras"))
decoder = tf.keras.models.load_model(os.path.join(BASE_DIR, "decoder_model.keras"))
proto_normal = np.load(os.path.join(BASE_DIR, "proto_normal.npy"))

# Seed Projection Matrix for HDC
np.random.seed(42)
proj_matrix = np.random.randn(LATENT_DIM, HV_DIM)

def encode_hdc(z_vector):
    # Continuous to Bipolar Hyperdimensional Encoding
    projected = np.dot(z_vector, proj_matrix)
    return np.sign(projected)

def cosine_similarity(hv1, hv2):
    return np.dot(hv1, hv2) / (np.linalg.norm(hv1) * np.linalg.norm(hv2) + 1e-8)

def predict(input_img):
    if input_img is None:
        return None, "Please upload an image."

    # Preprocess Image
    img = Image.fromarray(input_img).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # 1. ConvVAE Pass
    z_mean, z_log_var = encoder.predict(img_batch, verbose=0)
    recon_img = decoder.predict(z_mean, verbose=0)[0]

    # 2. HDC Anomaly Scoring
    hv_sample = encode_hdc(z_mean[0])
    sim = cosine_similarity(hv_sample, proto_normal)
    anomaly_score = 1.0 - sim

    # Decision (Threshold = 0.53)
    threshold = 0.53
    is_aml = anomaly_score > threshold
    status = "AML DETECTED" if is_aml else "NORMAL"

    # Generate Reconstruction Heatmap
    diff = np.abs(img_array - recon_img)
    heatmap = np.mean(diff, axis=-1)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].imshow(img_array)
    ax[0].set_title("Input Cell")
    ax[0].axis("off")
    
    ax[1].imshow(recon_img)
    ax[1].set_title("Reconstruction")
    ax[1].axis("off")

    ax[2].imshow(heatmap, cmap="jet")
    ax[2].set_title(f"XAI Map (Score: {anomaly_score:.3f})")
    ax[2].axis("off")

    plt.tight_layout()
    plot_path = "/tmp/result.png"
    plt.savefig(plot_path)
    plt.close()

    results_text = f"""
    ### Diagnosis: {status}
    * **Anomaly Score:** `{anomaly_score:.4f}`
    * **Decision Threshold:** `{threshold}`
    * **HDC Cosine Similarity:** `{sim:.4f}`
    """
    
    return plot_path, results_text

# Gradio Interface Setup
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Blood Cell Image"),
    outputs=[gr.Image(label="Visual Analysis"), gr.Markdown(label="Results")],
    title="AML Anomaly Detection via ConvVAE + HDC",
    description="ConvVAE reconstruction paired with Hyperdimensional Computing (HDC) binary vectors for AML triage."
)

if __name__ == "__main__":
    interface.launch()