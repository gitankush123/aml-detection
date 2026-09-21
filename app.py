import os

import numpy as np

import tensorflow as tf

from tensorflow.keras import layers, Model

from PIL import Image

import matplotlib.pyplot as plt

import streamlit as st

import io



# =============================================================

# CONFIGURATION — must match Kaggle training exactly

# =============================================================

st.set_page_config(

    page_title="AML Anomaly Detection — ConvVAE + HDC",

    page_icon="🔬",

    layout="wide"

)



IMG_SIZE   = 160    # ✅ matches training

LATENT_DIM = 64     # ✅ matches training

HV_DIM     = 10000  # ✅ matches training

GRID_SIZE  = 16     # FIX 1: was 8, must be 16 to match training

PATCH_FEAT = 4      # ✅ matches training



# FIX 5: Correct threshold

# Training uses: ANOMALY if cosine_sim < 0.5728

# App uses anomaly_score = 1 - sim, so: ANOMALY if anomaly_score > (1 - 0.5728)

THRESHOLD = 0.4272   # FIX: was 0.597, correct is 1 - 0.5728 = 0.4272



# Fixed projection matrices — seed 42 must match training

np.random.seed(42)

RP_latent  = np.random.randn(LATENT_DIM, HV_DIM).astype(np.float32)

RP_heatmap = np.random.randn(PATCH_FEAT,  HV_DIM).astype(np.float32)



BASE_DIR     = os.path.dirname(os.path.abspath(__file__))

ENCODER_PATH = os.path.join(BASE_DIR, "encoder.weights.h5")

DECODER_PATH = os.path.join(BASE_DIR, "decoder.weights.h5")

PROTO_PATH   = os.path.join(BASE_DIR, "proto_normal.npy")



# =============================================================

# ARCHITECTURE — must match Kaggle build_encoder / build_decoder

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

# LOAD MODELS

# FIX 4: Export as .h5 weights (not .keras) and load with load_weights()

# =============================================================

@st.cache_resource

def load_pipeline():

    tf.config.set_visible_devices([], 'GPU')

    tf.keras.backend.clear_session()



    missing = [f for f in [ENCODER_PATH, DECODER_PATH, PROTO_PATH]

               if not os.path.exists(f)]

    if missing:

        st.error(f"Missing files: {missing}\nUpload them to your GitHub repo.")

        st.stop()



    enc   = build_encoder()

    dec   = build_decoder()

    enc.load_weights(ENCODER_PATH)

    dec.load_weights(DECODER_PATH)

    proto = np.load(PROTO_PATH).astype(np.float32)

    proto = np.sign(proto); proto[proto == 0] = 1.0

    return enc, dec, proto



encoder, decoder, proto_normal = load_pipeline()



# =============================================================

# HDC PIPELINE — must match Kaggle pipeline exactly

# =============================================================

def encode_to_hv(vec, RP):

    vec = vec / (np.linalg.norm(vec) + 1e-8)

    hv  = np.sign(vec @ RP).astype(np.float32)

    hv[hv == 0] = 1.0

    return hv



def get_hv_final(img_np):

    inp      = tf.constant(img_np[np.newaxis], dtype=tf.float32)

    z_m, _, _ = encoder(inp, training=False)

    recon    = decoder(z_m, training=False)

    z_np     = z_m.numpy()[0]

    recon_np = recon.numpy()[0]



    # Global branch

    hv_lat = encode_to_hv(z_np, RP_latent)



    # Local branch

    error   = (img_np - recon_np) ** 2

    heatmap = np.sqrt(np.max(error, axis=-1))  # sqrt sharpening

    ph      = IMG_SIZE // GRID_SIZE             # 160//16 = 10px per patch

    accum   = np.zeros(HV_DIM, dtype=np.float32)



    for r in range(GRID_SIZE):

        for c in range(GRID_SIZE):

            patch = heatmap[r*ph:(r+1)*ph, c*ph:(c+1)*ph]

            feat  = np.array([patch.mean(), patch.max(),

                              patch.std(), np.percentile(patch, 90)],

                             dtype=np.float32)

            # FIX 3: z-score normalise patch features

            feat  = (feat - feat.mean()) / (feat.std() + 1e-8)

            accum += encode_to_hv(feat, RP_heatmap)



    hv_heat = np.sign(accum).astype(np.float32)

    hv_heat[hv_heat == 0] = 1.0



    # FIX 2: Full bind + bundle fusion (not just binding)

    bound   = hv_lat * hv_heat

    bundled = bound + hv_lat + hv_heat

    hv_fin  = np.sign(bundled).astype(np.float32)

    hv_fin[hv_fin == 0] = 1.0



    sim = float(np.dot(hv_fin, proto_normal) /

                (np.linalg.norm(hv_fin) * np.linalg.norm(proto_normal) + 1e-8))

    sim = float(np.clip(sim, -1.0, 1.0))

    mse = float(np.mean((img_np - recon_np) ** 2))



    return recon_np, heatmap, sim, mse



# =============================================================

# UI

# =============================================================

st.title("🔬 AML Blood Cancer Detection")

st.markdown("""

**ConvVAE + HDC Unsupervised Anomaly Detection — Ankush Agarwal (MIT-WPU Pune)**  

Trained on 10,000 healthy cells · **Zero AML labels** · AUC 0.9828 · Recall 96.10%

""")



st.sidebar.header("Controls")

threshold_display = st.sidebar.slider(

    "Cosine Similarity Threshold",

    min_value=0.30, max_value=0.80,

    value=0.5728, step=0.005,

    help="sim < threshold → ANOMALY"

)

st.sidebar.markdown(f"""

**Your trained system:**  

Normal mean sim ≈ **0.6069**  

AML mean sim ≈ **0.5357**  

Threshold **{threshold_display:.4f}** sits between them

""")



uploaded = st.sidebar.file_uploader(

    "Upload Blood Cell Image",

    type=["jpg","jpeg","png","tiff","tif","bmp"]

)



if uploaded is None:

    st.info("Upload a blood cell image from the sidebar to begin.")

else:

    pil_img = Image.open(uploaded).convert("RGB").resize(

        (IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    img_np  = np.array(pil_img, dtype=np.float32) / 255.0



    with st.spinner("Running ConvVAE + HDC pipeline..."):

        recon_np, heatmap, sim, mse = get_hv_final(img_np)



    # Use the slider threshold (cosine sim directly)

    is_anomaly = sim < threshold_display

    label      = "ANOMALY — AML BLAST DETECTED" if is_anomaly else "NORMAL"



    if is_anomaly:

        st.error(f"⚠️  {label}")

    else:

        st.success(f"✅  {label}")



    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Cosine Similarity", f"{sim:.4f}",

              delta=f"{sim - threshold_display:+.4f} vs threshold",

              delta_color="normal" if not is_anomaly else "inverse")

    c2.metric("Threshold",   f"{threshold_display:.4f}")

    c3.metric("Recon MSE",   f"{mse:.5f}")

    c4.metric("Anomaly Score", f"{1-sim:.4f}")



    st.divider()

    st.subheader("Explainable AI Output")

    col1, col2, col3 = st.columns(3)



    h_norm = (heatmap - heatmap.min()) / (heatmap.max()-heatmap.min()+1e-8)



    with col1:

        st.image(img_np, caption="Input Image", use_container_width=True)

    with col2:

        st.image(np.clip(recon_np,0,1),

                 caption="ConvVAE Reconstruction", use_container_width=True)

    with col3:

        fig, ax = plt.subplots(figsize=(4,4))

        ax.imshow(h_norm, cmap="hot", vmin=0, vmax=1)

        H, W = h_norm.shape

        for r in range(1, GRID_SIZE):

            ax.axhline(r*H//GRID_SIZE, color="white", lw=0.3, alpha=0.5)

        for c in range(1, GRID_SIZE):

            ax.axvline(c*W//GRID_SIZE, color="white", lw=0.3, alpha=0.5)

        ax.set_title(f"Anomaly Heatmap\n({GRID_SIZE}×{GRID_SIZE} grid)", fontsize=9)

        ax.axis("off")

        buf = io.BytesIO()

        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")

        plt.close()

        st.image(buf, caption="Level 1 — Pixel Heatmap", use_container_width=True)



    st.markdown(f"""

| Feature | Value |

|---|---|

| Cosine similarity | `{sim:.4f}` |

| Classification threshold | `{threshold_display:.4f}` |

| Decision | **{label}** |

| Reconstruction MSE | `{mse:.5f}` |

| GRID_SIZE | `{GRID_SIZE}×{GRID_SIZE}` = {GRID_SIZE*GRID_SIZE} patches |

| HD_DIM | `{HV_DIM}` |

    """)



st.divider()

st.caption("ConvVAE + HDC · MIT-WPU Pune · AUC 0.9828 · Zero AML Training Labels")
