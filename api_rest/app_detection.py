import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import requests
import base64
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🔍 Object Detection – Roboflow",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    h1, h2, h3 { color: #00d4ff; }
    .metric-box {
        background: #1e2130;
        border: 1px solid #00d4ff33;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #00d4ff; }
    .metric-label { font-size: 0.85rem; color: #aaa; }
    .detection-badge {
        display: inline-block;
        background: #00d4ff22;
        border: 1px solid #00d4ff55;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 3px;
        font-size: 0.85rem;
        color: #00d4ff;
    }
    div[data-testid="stSidebar"] { background-color: #161b27; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PALETTE
# ─────────────────────────────────────────────
PALETTE = [
    "#FF4B4B","#00D4FF","#FFD700","#00FF88",
    "#FF69B4","#FF8C00","#7B68EE","#00FA9A",
]

# ─────────────────────────────────────────────
#  SIDEBAR – CONFIGURATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration Roboflow")
    st.markdown("---")

    st.markdown("""
**Comment trouver le Model ID ?**
1. Va sur [roboflow.com](https://roboflow.com)
2. Ouvre ton projet → onglet **Deploy**
3. Le Model ID est affiché sous la forme :
   `nom-du-projet-xxxxx`
4. La version est le numéro (ex: `1`)
    """)
    st.markdown("---")

    api_key    = st.text_input("🔑 API Key", type="password",
                               placeholder="votre-cle-api-roboflow")
    model_id   = st.text_input("📁 Model ID (sans espaces)",
                               placeholder="objet_detection-3mzud",
                               help="Ex: objet_detection-3mzud  —  PAS d'espace, PAS de slash")
    version    = st.text_input("🔢 Version", value="1")
    confidence = st.slider("🎯 Confiance min (%)", 10, 100, 40)
    overlap    = st.slider("🔀 Chevauchement max (%)", 0, 100, 30)

    # Live URL preview
    if model_id and version:
        clean_id = model_id.strip().replace(" ", "-")
        st.markdown("**URL qui sera utilisée :**")
        st.code(
            f"https://detect.roboflow.com/{clean_id}/{version.strip()}"
            f"?api_key=***&confidence={confidence}&overlap={overlap}",
            language="text"
        )


# ─────────────────────────────────────────────
#  ROBOFLOW INFERENCE  ← URL CORRIGÉE
# ─────────────────────────────────────────────
def run_inference(image_pil: Image.Image) -> dict | None:
    """
    URL correcte Roboflow :
      POST https://detect.roboflow.com/{model_id}/{version}?api_key=...
    ⚠️  PAS de workspace dans l'URL
    ⚠️  PAS d'espace dans model_id
    """
    if not all([api_key, model_id, version]):
        st.warning("⚠️ Renseigne la clé API, le Model ID et la version dans la sidebar.")
        return None

    clean_id = model_id.strip().replace(" ", "-")

    # Encode image en base64
    buf = io.BytesIO()
    image_pil.convert("RGB").save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    url = (
        f"https://detect.roboflow.com/{clean_id}/{version.strip()}"
        f"?api_key={api_key.strip()}"
        f"&confidence={confidence}"
        f"&overlap={overlap}"
        f"&format=json"
    )

    try:
        resp = requests.post(
            url,
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        if resp.status_code != 200:
            st.error(
                f"❌ Erreur Roboflow {resp.status_code}\n\n"
                f"Réponse : `{resp.text[:300]}`\n\n"
                f"URL testée : `https://detect.roboflow.com/{clean_id}/{version.strip()}`"
            )
            return None
        return resp.json()

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur réseau : {e}")
        return None


# ─────────────────────────────────────────────
#  DRAW BOXES
# ─────────────────────────────────────────────
def draw_boxes(image_pil: Image.Image, predictions: list) -> Image.Image:
    img   = image_pil.copy().convert("RGB")
    draw  = ImageDraw.Draw(img)
    classes = list({p["class"] for p in predictions})

    for pred in predictions:
        color = PALETTE[classes.index(pred["class"]) % len(PALETTE)]
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x0, y0, x1, y1 = x - w/2, y - h/2, x + w/2, y + h/2

        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{pred['class']} {pred['confidence']*100:.0f}%"
        tw    = draw.textlength(label)
        draw.rectangle([x0, y0 - 18, x0 + tw + 8, y0], fill=color)
        draw.text((x0 + 4, y0 - 16), label, fill="black")

    return img


# ─────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────
def show_stats(predictions: list):
    if not predictions:
        st.info("Aucun objet détecté avec ces paramètres (essaie de baisser le seuil de confiance).")
        return

    classes = {}
    for p in predictions:
        classes[p["class"]] = classes.get(p["class"], 0) + 1

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{len(predictions)}</div>'
            f'<div class="metric-label">Objets détectés</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{len(classes)}</div>'
            f'<div class="metric-label">Classes distinctes</div></div>',
            unsafe_allow_html=True)
    with c3:
        avg = np.mean([p["confidence"] for p in predictions]) * 100
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-value">{avg:.1f}%</div>'
            f'<div class="metric-label">Confiance moyenne</div></div>',
            unsafe_allow_html=True)

    badges = " ".join(
        f'<span class="detection-badge">{cls}: {n}</span>'
        for cls, n in sorted(classes.items(), key=lambda x: -x[1])
    )
    st.markdown(badges, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
st.title("🔍 Détection d'Objets – Roboflow")
st.markdown("*Image unique · Groupe d'images · Webcam temps réel*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🖼️  Image unique",
    "📂  Groupe d'images",
    "📹  Webcam temps réel",
])

# ══════════════════════════════════════════════
#  TAB 1 – Image unique
# ══════════════════════════════════════════════
with tab1:
    st.header("Détection sur une image")
    uploaded = st.file_uploader(
        "Charger une image", type=["jpg","jpeg","png","bmp","webp"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Image originale")
            st.image(image, use_container_width=True)

        if st.button("🚀 Lancer la détection", key="single"):
            with st.spinner("Envoi à Roboflow…"):
                result = run_inference(image)

            if result and "predictions" in result:
                preds     = result["predictions"]
                annotated = draw_boxes(image, preds)
                with col2:
                    st.subheader(f"Résultats — {len(preds)} objet(s)")
                    st.image(annotated, use_container_width=True)
                st.markdown("---")
                show_stats(preds)

                buf = io.BytesIO()
                annotated.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Télécharger le résultat",
                    buf.getvalue(), "detection.png", "image/png")


# ══════════════════════════════════════════════
#  TAB 2 – Batch
# ══════════════════════════════════════════════
with tab2:
    st.header("Détection sur un groupe d'images")
    files = st.file_uploader(
        "Charger plusieurs images",
        type=["jpg","jpeg","png","bmp","webp"],
        accept_multiple_files=True,
        key="batch")

    if files and st.button("🚀 Analyser tout le groupe", key="batch_run"):
        all_preds = []
        bar = st.progress(0, "Analyse en cours…")

        for i, f in enumerate(files):
            img    = Image.open(f).convert("RGB")
            result = run_inference(img)
            preds  = result.get("predictions", []) if result else []
            all_preds.extend(preds)

            annotated = draw_boxes(img, preds)
            with st.expander(f"📄 {f.name}  —  {len(preds)} objet(s)", expanded=False):
                a, b = st.columns(2)
                a.image(img,       caption="Original",   use_container_width=True)
                b.image(annotated, caption="Détections", use_container_width=True)
                show_stats(preds)

            bar.progress((i+1)/len(files), f"Image {i+1}/{len(files)}")

        st.markdown("---")
        st.subheader("📊 Statistiques globales du groupe")
        show_stats(all_preds)
        bar.empty()


# ══════════════════════════════════════════════
#  TAB 3 – Webcam
# ══════════════════════════════════════════════
with tab3:
    st.header("Détection en temps réel – Webcam")
    st.info(
        "📸 Prends une photo avec la webcam → elle est envoyée à Roboflow → "
        "les objets sont annotés instantanément.")

    cam = st.camera_input("Prendre une photo")

    if cam:
        img = Image.open(cam).convert("RGB")
        with st.spinner("Détection en cours…"):
            result = run_inference(img)

        if result and "predictions" in result:
            preds     = result["predictions"]
            annotated = draw_boxes(img, preds)

            c1, c2 = st.columns(2)
            c1.image(img,       caption="Capture brute",          use_container_width=True)
            c2.image(annotated, caption=f"{len(preds)} objet(s)", use_container_width=True)

            st.markdown("---")
            show_stats(preds)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#555;font-size:.8rem;'>"
    "Object Detection App · Roboflow Hosted Inference API · Streamlit</p>",
    unsafe_allow_html=True)
