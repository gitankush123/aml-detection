import os
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

# =============================================================
# STREAMLIT PAGE CONFIG
# =============================================================
st.set_page_config(page_title="AML Cell Anomaly Detector", layout="centered")
st.title("AML Anomaly Detection via ConvVAE + HDC")
st.write("ConvVAE reconstruction paired with Hyperdimensional Computing (HDC) binary vectors for AML triage.")

# =============================================================
# CONFIGURATION & CONSTANTS
# =============================================================
IMG_SIZE   = 160
LATENT_DIM = 64
HV_DIM     = 10000
GRID_SIZE  = 16
PATCH_FEAT = 4
THRESHOLD  = 0.597  # Recalibrated threshold range [0.35 - 0.50]

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.weights.h5")
DECODER_PATH = os.path.join(BASE_DIR, "decoder.weights.h5")
PROTO_PATH   = os.path.join(BASE_DIR, "proto_normal.npy")

ENCODER_URL = "https://github.com/gitankush123/aml-detection/releases/download/v1.0.0/encoder.weights.h5"
DECODER_URL = "https://github.com/gitankush123/aml-detection/releases/download/v1.0.0/decoder.weights.h5"

# Fixed random state for projection matrices
np.random.seed(42)
RP_latent  = np.random.randn(LATENT_DIM, HV_DIM).astype(np.float32)
RP_heatmap = np.random.randn(PATCH_FEAT, HV_DIM).astype(np.float32)

# =============================================================
# 1. WEIGHT DOWNLOADER
# =============================================================
@st.cache_resource
def download_weights_if_missing():
    if not os.path.exists(ENCODER_PATH):
        st.info("Downloading encoder weights from GitHub Release...")
        urllib.request.urlretrieve(ENCODER_URL, ENCODER_PATH)
        
    if not os.path.exists(DECODER_PATH):
        st.info("Downloading decoder weights from GitHub Release...")
        urllib.request.urlretrieve(DECODER_URL, DECODER_PATH)

download_weights_if_missing()

# =============================================================
# 2. ARCHITECTURE DEFINITIONS & MODEL LOADING
# =============================================================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def build_encoder():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = layers.Conv2D(32,  3, strides=2, padding="same", activation="relu")(inp)
    x   = layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")(x)
    x   = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x   = layers.Flatten()(x)
    x   = layers.Dense(256, activation="relu")(x)
    z_mean    = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z         = Sampling()([z_mean, z_log_var])
    return Model(inp, [z_mean, z_log_var, z], name="encoder")

def build_decoder():
    inp = layers.Input(shape=(LATENT_DIM,))
    x   = layers.Dense((IMG_SIZE // 8) * (IMG_SIZE // 8) * 128, activation="relu")(inp)
    x   = layers.Reshape((IMG_SIZE // 8, IMG_SIZE // 8, 128))(x)
    x   = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x   = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")(x)
    x   = layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2DTranspose(3,   3, padding="same", activation="sigmoid")(x)
    return Model(inp, out, name="decoder")

@st.cache_resource
def load_full_pipeline():
    enc = build_encoder()
    dec = build_decoder()
    enc.load_weights(ENCODER_PATH)
    dec.load_weights(DECODER_PATH)
    proto = np.load(PROTO_PATH)
    return enc, dec, proto

encoder, decoder, proto_normal = load_full_pipeline()

# =============================================================
# 3. HDC PIPELINE UTILITIES
# =============================================================
def binarize_hv(proj):
    """Maps projection arrays strictly to {-1.0, 1.0}."""
    hv = np.sign(proj).astype(np.float32)
    hv[hv == 0] = 1.0
    return hv

def encode_to_hv(vec, RP):
    norm = np.linalg.norm(vec) + 1e-8
    vec_norm = vec / norm
    proj = vec_norm @ RP
    return binarize_hv(proj)

def extract_latent_hv(z_vec):
    return encode_to_hv(z_vec, RP_latent)

def extract_heatmap_hv(img_array, recon_img):
    error   = (img_array - recon_img) ** 2
    heatmap = np.sqrt(np.max(error, axis=-1))

    H, W = heatmap.shape
    ph, pw = H // GRID_SIZE, W // GRID_SIZE
    hv_accum = np.zeros(HV_DIM, dtype=np.float32)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            patch = heatmap[r*ph:(r+1)*ph, c*pw:(c+1)*pw]
            feat  = np.array([
                np.mean(patch),
                np.max(patch),
                np.std(patch),
                np.percentile(patch, 90)
            ], dtype=np.float32)

            hv_accum += encode_to_hv(feat, RP_heatmap)

    return binarize_hv(hv_accum)

def fuse_hvs(hv1, hv2):
    """HDC Vector Symbolic Binding via Element-wise Multiplication."""
    return hv1 * hv2

def cosine_similarity(hv1, hv2):
    dot_prod = np.dot(hv1, hv2)
    norms = (np.linalg.norm(hv1) * np.linalg.norm(hv2)) + 1e-8
    return float(dot_prod / norms)

# =============================================================
# 4. STREAMLIT USER INTERFACE
# =============================================================
uploaded_file = st.file_uploader("Upload Blood Cell Image", type=["jpg", "png", "jpeg", "tiff"])

if uploaded_file is not None:
    # 1. Robust Image Preprocessing
    raw_img = Image.open(uploaded_file).convert("RGB")
    img_resized = raw_img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # 2. VAE Reconstruction Pass
    z_mean, _, _ = encoder.predict(img_batch, verbose=0)
    recon_img    = decoder.predict(z_mean, verbose=0)[0]

    # 3. Dual-Modal HDC Extraction & Fusion
    hv_lat  = extract_latent_hv(z_mean[0])
    hv_hm   = extract_heatmap_hv(img_array, recon_img)
    hv_fused = fuse_hvs(hv_lat, hv_hm)

    # 4. Anomaly Decision Logic
    sim = cosine_similarity(hv_fused, proto_normal)
    anomaly_score = 1.0 - sim
    is_aml = anomaly_score > THRESHOLD

    # 5. Visualizations
    diff = np.abs(img_array - recon_img)
    heatmap_viz = np.mean(diff, axis=-1)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].imshow(img_array)
    ax[0].set_title("Input Cell")
    ax[0].axis("off")
    
    ax[1].imshow(np.clip(recon_img, 0.0, 1.0))
    ax[1].set_title("Reconstruction")
    ax[1].axis("off")

    ax[2].imshow(heatmap_viz, cmap="jet")
    ax[2].set_title(f"XAI Map (Score: {anomaly_score:.3f})")
    ax[2].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    # 6. Diagnosis Outputs
    if is_aml:
        st.error("### Diagnosis: AML DETECTED")
    else:
        st.success("### Diagnosis: NORMAL")

    st.markdown(f"""
    * **Anomaly Score:** `{anomaly_score:.4f}`
    * **Decision Threshold:** `{THRESHOLD}`
    * **HDC Cosine Similarity:** `{sim:.4f}`
    """)
