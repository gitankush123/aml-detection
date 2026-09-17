import os
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

# =============================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# =============================================================
st.set_page_config(
    page_title="AML Anomaly Detection via ConvVAE-HDC",
    page_icon="🔬",
    layout="wide"
)

IMG_SIZE = 160
LATENT_DIM = 64
HD_DIM = 10000
GRID_SIZE = 16
PATCH_FEAT = 4

# Paths for local model artifacts
ENCODER_WEIGHTS = "encoder.weights.h5"
DECODER_WEIGHTS = "decoder.weights.h5"
PROTO_PATH = "proto_normal.npy"

# Replace these placeholder URLs with direct download links to your hosted artifacts
URL_ENCODER = "https://your-hosting-domain.com/path/to/encoder.weights.h5"
URL_DECODER = "https://your-hosting-domain.com/path/to/decoder.weights.h5"
URL_PROTO = "https://your-hosting-domain.com/path/to/proto_normal.npy"

# Fixed seed projection matrices matching training setup
np.random.seed(42)
RP_LATENT = np.random.randn(LATENT_DIM, HD_DIM).astype(np.float32)
RP_HEATMAP = np.random.randn(PATCH_FEAT, HD_DIM).astype(np.float32)

# Optimal decision threshold derived from ROC analysis
DEFAULT_SIM_THRESHOLD = 0.403  # Cosine similarity threshold (sim < thr => Anomaly)

# =============================================================
# 2. VAE ARCHITECTURE DEFINITIONS
# =============================================================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def build_encoder():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return Model(inp, [z_mean, z_log_var, z], name="encoder")

def build_decoder():
    inp = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense((IMG_SIZE // 8) * (IMG_SIZE // 8) * 128, activation="relu")(inp)
    x = layers.Reshape((IMG_SIZE // 8, IMG_SIZE // 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid")(x)
    return Model(inp, out, name="decoder")

# =============================================================
# 3. RESOURCE LOADING & CACHING
# =============================================================
def download_if_missing(file_path, url):
    """Helper function using urllib.request to fetch model files if not present locally."""
    if not os.path.exists(file_path):
        st.info(f"Downloading artifact `{file_path}` from remote storage...")
        try:
            urllib.request.urlretrieve(url, file_path)
            st.success(f"Successfully downloaded `{file_path}`.")
        except Exception as e:
            st.error(f"Failed to download `{file_path}` from {url}. Error: {e}")

@st.cache_resource
def load_pipeline():
    tf.keras.backend.clear_session()
    
    # Instantiate models
    encoder = build_encoder()
    decoder = build_decoder()
    
    # Download artifacts via urllib.request if they are not in local storage
    download_if_missing(ENCODER_WEIGHTS, URL_ENCODER)
    download_if_missing(DECODER_WEIGHTS, URL_DECODER)
    download_if_missing(PROTO_PATH, URL_PROTO)

    # Load weights if available
    if os.path.exists(ENCODER_WEIGHTS) and os.path.exists(DECODER_WEIGHTS):
        encoder.load_weights(ENCODER_WEIGHTS)
        decoder.load_weights(DECODER_WEIGHTS)
    else:
        st.warning("Model weights not found locally or via download. Running with uninitialized model parameters.")

    # Load normal hypervector prototype
    if os.path.exists(PROTO_PATH):
        proto_normal = np.load(PROTO_PATH)
        # Enforce strict bipolar normalization {-1.0, 1.0}
        proto_normal = np.sign(proto_normal).astype(np.float32)
        proto_normal[proto_normal == 0] = 1.0
    else:
        st.warning("`proto_normal.npy` not found. Generating dummy random prototype.")
        proto_normal = np.sign(np.random.randn(HD_DIM)).astype(np.float32)
        proto_normal[proto_normal == 0] = 1.0

    return encoder, decoder, proto_normal

encoder, decoder, proto_normal = load_pipeline()

# =============================================================
# 4. HDC ENCODING & SCORING FUNCTIONS
# =============================================================
def encode_to_hv(vec, RP):
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    proj = vec @ RP
    hv = np.sign(proj).astype(np.float32)
    hv[hv == 0] = 1.0
    return hv

def extract_fused_hv(img_np, encoder, decoder):
    # Single forward pass through VAE
    inp = tf.constant(img_np[np.newaxis], dtype=tf.float32)
    z_m, _, _ = encoder(inp, training=False)
    recon = decoder(z_m, training=False)
    
    z_np = z_m.numpy()[0]
    recon_np = recon.numpy()[0]

    # Global HV encoding
    hv_latent = encode_to_hv(z_np, RP_LATENT)

    # Reconstruction heatmap local feature aggregation
    error = (img_np - recon_np) ** 2
    heatmap = np.sqrt(np.max(error, axis=-1))

    H, W = heatmap.shape
    ph, pw = H // GRID_SIZE, W // GRID_SIZE
    hv_accum = np.zeros(HD_DIM, dtype=np.float32)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            patch = heatmap[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
            feat = np.array([
                np.mean(patch), np.max(patch),
                np.std(patch), np.percentile(patch, 90)
            ], dtype=np.float32)
            feat = (feat - feat.mean()) / (feat.std() + 1e-8)
            hv_accum += encode_to_hv(feat, RP_HEATMAP)

    hv_heat = np.sign(hv_accum).astype(np.float32)
    hv_heat[hv_heat == 0] = 1.0

    # Hypervector Fusion (Binding + Bundling)
    bound = hv_latent * hv_heat
    bundled = bound + hv_latent + hv_heat
    hv_fused = np.sign(bundled).astype(np.float32)
    hv_fused[hv_fused == 0] = 1.0

    return hv_fused, recon_np, heatmap

def compute_metrics(hv_fused, proto_normal):
    dot = np.dot(hv_fused, proto_normal)
    norm_sample = np.linalg.norm(hv_fused)
    norm_proto = np.linalg.norm(proto_normal)
    
    # Cosine Similarity bounded to [-1.0, 1.0]
    sim = float(dot / (norm_sample * norm_proto + 1e-8))
    sim = np.clip(sim, -1.0, 1.0)
    
    # Anomaly Score bounded to [0.0, 1.0]
    anomaly_score = max(0.0, min(1.0, 1.0 - sim))
    return sim, anomaly_score

# =============================================================
# 5. STREAMLIT USER INTERFACE
# =============================================================
st.title("🔬 Acute Myeloid Leukemia (AML) Blast Detection")
st.markdown("""
This application uses a combined **Convolutional Variational Autoencoder (ConvVAE)** and **Hyperdimensional Computing (HDC)** architecture to detect morphologic anomalies in blood cell microscopy images.
""")

st.sidebar.header("Configuration & Controls")
sim_threshold = st.sidebar.slider(
    "Cosine Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_SIM_THRESHOLD,
    step=0.01,
    help="Samples with similarity BELOW this threshold are classified as AML ANOMALY."
)

anomaly_threshold = 1.0 - sim_threshold
st.sidebar.markdown(f"**Equivalent Anomaly Score Threshold:** `{anomaly_threshold:.3f}`")

uploaded_file = st.sidebar.file_uploader("Upload Peripheral Blood Smear Image", type=["jpg", "png", "tif", "tiff"])

if uploaded_file is not None:
    # Preprocess image
    pil_img = Image.open(uploaded_file).convert("RGB")
    pil_img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_np = np.array(pil_img_resized, dtype=np.float32) / 255.0

    # Run inference
    with st.spinner("Processing image through ConvVAE-HDC pipeline..."):
        hv_fused, recon_np, heatmap = extract_fused_hv(img_np, encoder, decoder)
        sim, anomaly_score = compute_metrics(hv_fused, proto_normal)

    # Classification decision
    is_anomaly = sim < sim_threshold
    status_label = "AML BLAST DETECTED (ANOMALY)" if is_anomaly else "NORMAL CELL"
    status_color = "red" if is_anomaly else "green"

    # Top Results Banner
    st.subheader("Diagnostic Results")
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.metric("Cosine Similarity", f"{sim:.4f}")
    with res_col2:
        st.metric("Anomaly Score (1 - Sim)", f"{anomaly_score:.4f}")
    with res_col3:
        st.markdown(f"### Classification:\n**:<font color='{status_color}'>{status_label}</font>**", unsafe_allow_html=True)

    st.markdown("---")

    # Visualizations
    st.subheader("Explainable AI (XAI) Reconstruction & Heatmap")
    viz_col1, viz_col2, viz_col3 = st.columns(3)

    # Normalize heatmap for visualization
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    with viz_col1:
        st.image(img_np, caption="Input Image (160x160)", use_container_width=True)
    with viz_col2:
        st.image(np.clip(recon_np, 0, 1), caption="VAE Reconstruction", use_container_width=True)
    with viz_col3:
        fig, ax = plt.subplots()
        ax.imshow(heatmap_norm, cmap="hot")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Technical Details Debug Expander
    with st.expander("Show Technical Feature Details"):
        st.write(f"**Fused Hypervector Dimension:** `{hv_fused.shape[0]}`")
        st.write(f"**Sample Vector Norm:** `{np.linalg.norm(hv_fused):.2f}`")
        st.write(f"**Prototype Vector Norm:** `{np.linalg.norm(proto_normal):.2f}`")
        st.write(f"**Dot Product:** `{np.dot(hv_fused, proto_normal):.2f}`")
else:
    st.info("Please upload an image from the sidebar to begin analysis.")
