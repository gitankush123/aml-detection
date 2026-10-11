import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import io

st.set_page_config(
    page_title="AML Detection — ConvVAE + HDC",
    page_icon="🔬",
    layout="wide"
)

# =============================================================
# CONFIG — must match Kaggle training exactly
# =============================================================
IMG_SIZE   = 160
LATENT_DIM = 64
HD_DIM     = 10000
GRID_SIZE  = 16
PATCH_FEAT = 4
THRESHOLD  = 0.5728

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
ENCODER_PATH    = os.path.join(BASE_DIR, "encoder.weights.h5")
DECODER_PATH    = os.path.join(BASE_DIR, "decoder.weights.h5")
PROTO_PATH      = os.path.join(BASE_DIR, "proto_normal.npy")
RP_LATENT_PATH  = os.path.join(BASE_DIR, "RP_latent.npy")
RP_HEATMAP_PATH = os.path.join(BASE_DIR, "RP_heatmap.npy")

# =============================================================
# ARCHITECTURE
# =============================================================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def build_encoder():
    inp       = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x         = layers.Conv2D(32,  3, strides=2, padding="same", activation="relu")(inp)
    x         = layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")(x)
    x         = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x         = layers.Flatten()(x)
    x         = layers.Dense(256, activation="relu")(x)
    z_mean    = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z         = Sampling()([z_mean, z_log_var])
    return Model(inp, [z_mean, z_log_var, z], name="encoder")

def build_decoder():
    inp = layers.Input(shape=(LATENT_DIM,))
    x   = layers.Dense((IMG_SIZE//8)*(IMG_SIZE//8)*128, activation="relu")(inp)
    x   = layers.Reshape((IMG_SIZE//8, IMG_SIZE//8, 128))(x)
    x   = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x   = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")(x)
    x   = layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2DTranspose(3,   3, padding="same", activation="sigmoid")(x)
    return Model(inp, out, name="decoder")

# =============================================================
# LOAD PIPELINE — matrices loaded from saved files, not regenerated
# =============================================================
@st.cache_resource
def load_pipeline():
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.backend.clear_session()

    required = {
        "encoder.weights.h5": ENCODER_PATH,
        "decoder.weights.h5": DECODER_PATH,
        "proto_normal.npy":   PROTO_PATH,
        "RP_latent.npy":      RP_LATENT_PATH,
        "RP_heatmap.npy":     RP_HEATMAP_PATH,
    }
    missing = [name for name, path in required.items()
               if not os.path.exists(path)]
    if missing:
        st.error(f"Missing files in repository: {missing}\n"
                 f"Download from Kaggle working directory and upload to GitHub.")
        st.stop()

    enc = build_encoder()
    dec = build_decoder()
    enc.load_weights(ENCODER_PATH)
    dec.load_weights(DECODER_PATH)

    proto      = np.load(PROTO_PATH).astype(np.float32)
    proto      = np.sign(proto); proto[proto == 0] = 1.0

    # Load exact matrices from Kaggle — never regenerate
    RP_lat     = np.load(RP_LATENT_PATH).astype(np.float32)
    RP_heat    = np.load(RP_HEATMAP_PATH).astype(np.float32)

    # Verification
    assert RP_lat.shape  == (LATENT_DIM, HD_DIM), \
        f"RP_latent shape mismatch: {RP_lat.shape}"
    assert RP_heat.shape == (PATCH_FEAT,  HD_DIM), \
        f"RP_heatmap shape mismatch: {RP_heat.shape}"

    return enc, dec, proto, RP_lat, RP_heat

encoder, decoder, proto_normal, RP_latent, RP_heatmap = load_pipeline()

st.sidebar.write("Prototype shape:", proto_normal.shape)
st.sidebar.write("Prototype unique:", np.unique(proto_normal))
st.sidebar.write("Prototype norm:", np.linalg.norm(proto_normal))

st.sidebar.write("RP_latent first 5:", RP_latent[0, :5])
st.sidebar.write("RP_heatmap first 5:", RP_heatmap[0, :5])
st.sidebar.write("Prototype first 10:", proto_normal[:10])

assert proto_normal.shape == (HD_DIM,)
assert RP_latent.shape == (LATENT_DIM, HD_DIM)
assert RP_heatmap.shape == (PATCH_FEAT, HD_DIM)

# Debug mode
debug = st.sidebar.checkbox("Show debug info", value=False)

# =============================================================
# HDC PIPELINE
# =============================================================
def encode_to_hv(vec, RP):
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    hv  = np.sign(vec @ RP).astype(np.float32)
    hv[hv == 0] = 1.0
    return hv

def cosine_sim(a, b):
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def run_pipeline(img_np):
    inp       = tf.constant(img_np[np.newaxis], dtype=tf.float32)
    z_m, _, _ = encoder(inp, training=False)
    recon     = decoder(z_m, training=False)
    z_np      = z_m.numpy()[0]
    recon_np  = recon.numpy()[0]

    # Global branch
    hv_lat  = encode_to_hv(z_np, RP_latent)
    sim_latent = cosine_sim(hv_lat, proto_normal)

    # Local branch — 16×16 grid
    error   = (img_np - recon_np) ** 2
    heatmap = np.sqrt(np.max(error, axis=-1))
    ph      = IMG_SIZE // GRID_SIZE
    accum   = np.zeros(HD_DIM, dtype=np.float32)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            patch = heatmap[r*ph:(r+1)*ph, c*ph:(c+1)*ph]
            feat  = np.array([patch.mean(), patch.max(),
                              patch.std(), np.percentile(patch, 90)],
                             dtype=np.float32)
            feat  = (feat - feat.mean()) / (feat.std() + 1e-8)
            accum += encode_to_hv(feat, RP_heatmap)

    hv_heat = np.sign(accum).astype(np.float32)
    hv_heat[hv_heat == 0] = 1.0
    sim_heat = cosine_sim(hv_heat, proto_normal)

    # Fusion — bind + bundle
    bound   = hv_lat * hv_heat
    hv_fin  = np.sign(bound + hv_lat + hv_heat).astype(np.float32)
    hv_fin[hv_fin == 0] = 1.0
    sim_final = cosine_sim(hv_fin, proto_normal)

    if debug:
        st.write({
            "latent_similarity": sim_latent,
            "heatmap_similarity": sim_heat,
            "final_similarity": sim_final,
            "latent_hv_unique": np.unique(hv_lat),
            "heat_hv_unique": np.unique(hv_heat),
            "final_hv_unique": np.unique(hv_fin),
        })

    sim = float(np.clip(cosine_sim(hv_fin, proto_normal), -1.0, 1.0))
    mse = float(np.mean((img_np - recon_np) ** 2))

    return recon_np, heatmap, sim, mse

# =============================================================
# UI
# =============================================================
st.title("🔬 AML Blood Cancer Detection")
st.markdown("""
**ConvVAE + HDC Unsupervised Anomaly Detection — MIT-WPU Pune**
Trained on **10,000 healthy cells only** · Zero AML labels · AUC **0.9828** · Recall **96.10%**
""")

# Sidebar
st.sidebar.header("Controls")
threshold = st.sidebar.slider(
    "Cosine Similarity Threshold",
    min_value=0.40, max_value=0.75,
    value=THRESHOLD, step=0.005,
    help="sim < threshold → ANOMALY   |   sim ≥ threshold → NORMAL"
)
st.sidebar.markdown(f"""
**Trained system reference ranges:**
- Normal cells: sim ≈ **0.58 – 0.65**
- AML blast cells: sim ≈ **0.50 – 0.54**
- Current threshold: **{threshold:.4f}**
""")

uploaded = st.sidebar.file_uploader(
    "Upload Blood Cell Image",
    type=["jpg", "jpeg", "png", "tiff", "tif", "bmp"]
)

if debug:
    st.sidebar.markdown("### Debug")
    st.sidebar.write(f"RP_latent  shape: {RP_latent.shape}")
    st.sidebar.write(f"RP_heatmap shape: {RP_heatmap.shape}")
    st.sidebar.write(f"Proto_normal norm: {np.linalg.norm(proto_normal):.2f}")
    st.sidebar.write(f"Proto_normal first 5: {proto_normal[:5]}")

if uploaded is None:
    st.info("Upload a blood cell microscopy image from the sidebar to begin.")
    st.markdown("""
### How It Works
1. Image → ConvVAE encoder → **z_mean** (64-dim latent vector)
2. Decoder reconstructs image from z_mean → pixel error → **heatmap**
3. **Global branch:** z_mean → L2 norm → RP_latent → HV_latent
4. **Local branch:** heatmap → 16×16 patches → features → RP_heatmap → HV_heatmap
5. **Fusion:** HV_final = sign(HV_latent ⊙ HV_heatmap + HV_latent + HV_heatmap)
6. **cos_sim(HV_final, Proto_normal)** → classification decision
    """)
else:
    pil_img = Image.open(uploaded).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_np  = np.array(pil_img, dtype=np.float32) / 255.0

    with st.spinner("Running ConvVAE + HDC pipeline..."):
        recon_np, heatmap, sim, mse = run_pipeline(img_np)

    is_anomaly = sim < threshold
    label      = "ANOMALY — AML BLAST DETECTED" if is_anomaly else "NORMAL"

    if is_anomaly:
        st.error(f"⚠️  {label}")
    else:
        st.success(f"✅  {label}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cosine Similarity", f"{sim:.4f}",
              delta=f"{sim - threshold:+.4f} vs threshold",
              delta_color="normal" if not is_anomaly else "inverse")
    c2.metric("Threshold",    f"{threshold:.4f}")
    c3.metric("Recon MSE",    f"{mse:.5f}")
    c4.metric("Anomaly Score",f"{1-sim:.4f}")

    # Visual XAI output
    st.divider()
    st.subheader("Explainable AI — Three-Level Output")
    col1, col2, col3 = st.columns(3)

    h_norm = (heatmap - heatmap.min()) / (heatmap.max()-heatmap.min()+1e-8)

    with col1:
        st.image(img_np, caption="Input Image", use_container_width=True)
    with col2:
        st.image(np.clip(recon_np, 0, 1),
                 caption="ConvVAE Reconstruction (from z_mean)",
                 use_container_width=True)
    with col3:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(h_norm, cmap="hot", vmin=0, vmax=1)
        H, W = h_norm.shape
        for r in range(1, GRID_SIZE):
            ax.axhline(r*H//GRID_SIZE, color="white", lw=0.3, alpha=0.4)
        for c in range(1, GRID_SIZE):
            ax.axvline(c*W//GRID_SIZE, color="white", lw=0.3, alpha=0.4)
        ax.set_title(f"Level 1: Anomaly Heatmap\n"
                     f"({GRID_SIZE}×{GRID_SIZE} grid overlay)", fontsize=9)
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        st.image(buf, caption="Level 1 — Pixel Heatmap",
                 use_container_width=True)

    st.markdown(f"""
| Feature | Value | Interpretation |
|---|---|---|
| Cosine similarity | `{sim:.4f}` | How close HV_final is to Proto_normal |
| Threshold | `{threshold:.4f}` | Youden-optimal boundary |
| Decision | **{label}** | sim {"≥" if not is_anomaly else "<"} threshold |
| Normal ref range | `0.58 – 0.65` | Expected for healthy cells |
| AML ref range | `0.50 – 0.54` | Expected for blast cells |
| MSE | `{mse:.5f}` | Pixel reconstruction error |
    """)

    if debug:
        st.markdown("### Debug — Pipeline Internals")
        st.write(f"RP_latent first 5 values: {RP_latent[0,:5]}")
        st.write(f"RP_heatmap first 5 values: {RP_heatmap[0,:5]}")
        st.write(f"Proto_normal first 5 values: {proto_normal[:5]}")

st.divider()
st.caption("ConvVAE + HDC · MIT-WPU Pune · AUC 0.9828 · Recall 96.10% · Zero AML Training Labels")
