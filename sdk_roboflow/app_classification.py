import streamlit as st
import tempfile
import os
import json
import io
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────
st.set_page_config(page_title="Classification App", page_icon="🔍", layout="wide")
st.title("🔍 Classification App — Roboflow")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key       = st.text_input("Clé API Roboflow", type="password")
    project_id    = st.text_input("Project ID", value="cv_lab_roboflow-9fbpq-vrod5")
    model_version = st.number_input("Version", min_value=1, value=1)
    top_k         = st.slider("Top-K résultats", 1, 10, 3)

    if st.button("🚀 Charger le modèle", use_container_width=True):
        if not api_key:
            st.error("Clé API requise !")
        else:
            with st.spinner("Chargement..."):
                try:
                    from roboflow import Roboflow
                    rf = Roboflow(api_key=api_key)
                    model = rf.workspace().project(project_id).version(model_version).model
                    st.session_state.model = model
                    st.success("✅ Modèle chargé !")
                except Exception as e:
                    st.error(f"Erreur : {e}")

# ─────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────
def predict(model, img_bytes, filename):
    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name
    try:
        result = model.predict(tmp_path).json()
        return result
    finally:
        os.unlink(tmp_path)

def parse_predictions(result):
    preds = []
    if "predictions" in result and isinstance(result["predictions"], dict):
        for cls, conf in result["predictions"].items():
            preds.append({"class": cls, "confidence": conf})
        preds.sort(key=lambda x: x["confidence"], reverse=True)
    elif "predictions" in result and isinstance(result["predictions"], list):
        preds = result["predictions"]
        for p in preds:
            if "class" not in p:
                p["class"] = p.get("class_name", p.get("top", "?"))
        preds.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    elif "top" in result:
        preds = [{"class": result["top"], "confidence": result.get("confidence", 0)}]
    return preds

def draw_result(img_bytes, preds, top_k):
    img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    y = 10
    for i, p in enumerate(preds[:top_k]):
        cls  = p.get("class", "?")
        conf = p.get("confidence", 0) * 100
        txt  = f"#{i+1} {cls}  {conf:.1f}%"
        w    = len(txt) * 9 + 10
        draw.rectangle([8, y, 8+w, y+26], fill=(0, 0, 0))
        draw.text((12, y+4), txt, fill=(100, 220, 100))
        y += 32
    return img

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🖼️ Image", "📁 Dossier", "🎥 Vidéo"])

# ══════════════════
# TAB 1 — IMAGE
# ══════════════════
with tab1:
    uploaded = st.file_uploader("Choisir une image", type=["jpg","jpeg","png","bmp","webp"])

    if uploaded:
        if "model" not in st.session_state:
            st.warning("⚠️ Chargez d'abord le modèle dans la sidebar.")
        else:
            img_bytes = uploaded.read()

            with st.spinner("Classification en cours..."):
                result = predict(st.session_state.model, img_bytes, uploaded.name)
                preds  = parse_predictions(result)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🖼️ Image originale")
                st.image(img_bytes, use_column_width=True)
            with col2:
                st.subheader("🎯 Image annotée")
                ann = draw_result(img_bytes, preds, top_k)
                st.image(ann, use_column_width=True)

            st.markdown("---")
            st.subheader("📊 Prédictions")

            if not preds:
                st.error("❌ Aucune prédiction. Réponse brute :")
                st.json(result)
            else:
                for i, p in enumerate(preds[:top_k]):
                    cls  = p.get("class", "?")
                    conf = p.get("confidence", 0) * 100
                    col_a, col_b = st.columns([3, 1])
                    col_a.progress(int(conf), text=f"#{i+1}  **{cls}**")
                    col_b.metric("", f"{conf:.1f}%")
                st.success(f"✅ Résultat : **{preds[0].get('class','?')}**  —  {preds[0].get('confidence',0)*100:.1f}%")

            st.download_button(
                "⬇️ Télécharger JSON",
                json.dumps({"file": uploaded.name, "raw": result, "parsed": preds}, indent=2),
                file_name="classification.json",
                mime="application/json"
            )

# ══════════════════
# TAB 2 — DOSSIER
# ══════════════════
with tab2:
    mode  = st.radio("Mode", ["Plusieurs images", "Archive ZIP"], horizontal=True)
    files = []

    if mode == "Plusieurs images":
        files = st.file_uploader("Images", type=["jpg","jpeg","png","bmp","webp"],
                                 accept_multiple_files=True, key="folder")
    else:
        zf = st.file_uploader("Archive ZIP", type=["zip"])
        if zf:
            with tempfile.TemporaryDirectory() as d:
                with zipfile.ZipFile(zf) as z: z.extractall(d)
                for p in Path(d).rglob("*"):
                    if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}:
                        buf = io.BytesIO(p.read_bytes())
                        buf.name = p.name
                        files.append(buf)
            st.success(f"{len(files)} images extraites")

    if files:
        if "model" not in st.session_state:
            st.warning("⚠️ Chargez d'abord le modèle dans la sidebar.")
        elif st.button("▶️ Lancer", use_container_width=True):
            all_results = []
            progress    = st.progress(0)
            cols        = st.columns(3)
            col_idx     = 0

            for i, f in enumerate(files):
                try:
                    img_bytes = f.read()
                    result    = predict(st.session_state.model, img_bytes, f.name)
                    preds     = parse_predictions(result)
                    ann       = draw_result(img_bytes, preds, top_k)
                    label     = f"{preds[0]['class']} ({preds[0]['confidence']*100:.0f}%)" if preds else "?"
                    cols[col_idx].image(ann, caption=f"{f.name}\n{label}", use_column_width=True)
                    col_idx   = (col_idx + 1) % 3
                    all_results.append({"file": f.name, "predictions": preds})
                except Exception as e:
                    all_results.append({"file": f.name, "error": str(e)})
                progress.progress((i+1)/len(files))

            st.success(f"✅ {len(files)} images traitées")
            st.download_button("⬇️ Exporter JSON", json.dumps(all_results, indent=2),
                               file_name="classification_dossier.json", mime="application/json")

# ══════════════════
# TAB 3 — VIDÉO
# ══════════════════
with tab3:
    video_file   = st.file_uploader("Choisir une vidéo", type=["mp4","avi","mov","mkv"])
    col_a, col_b = st.columns(2)
    frame_skip   = col_a.number_input("Analyser 1 frame sur N", min_value=1, value=5)
    max_frames   = col_b.number_input("Max frames", min_value=1, value=50)

    if video_file:
        st.video(video_file)

        if "model" not in st.session_state:
            st.warning("⚠️ Chargez d'abord le modèle dans la sidebar.")
        elif st.button("▶️ Analyser la vidéo", use_container_width=True):
            import cv2
            import numpy as np
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.read())
                vid_path = tmp.name

            cap          = cv2.VideoCapture(vid_path)
            fps          = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_idx    = 0
            analyzed     = 0
            all_results  = []
            class_counts = {}
            progress     = st.progress(0)
            status       = st.empty()
            preview      = st.empty()

            while cap.isOpened() and analyzed < max_frames:
                ret, frame = cap.read()
                if not ret: break
                if frame_idx % int(frame_skip) == 0:
                    img_pil   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    buf       = io.BytesIO()
                    img_pil.save(buf, format="JPEG")
                    img_bytes = buf.getvalue()

                    result = predict(st.session_state.model, img_bytes, "frame.jpg")
                    preds  = parse_predictions(result)
                    ann    = draw_result(img_bytes, preds, top_k)
                    preview.image(ann, caption=f"Frame {frame_idx} — t={frame_idx/fps:.1f}s", use_column_width=True)

                    top = preds[0] if preds else {}
                    cls = top.get("class", "?")
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                    all_results.append({"frame": frame_idx, "time_s": round(frame_idx/fps, 2), "predictions": preds})
                    analyzed += 1
                    progress.progress(min(analyzed/int(max_frames), 1.0))
                    status.text(f"Frame {frame_idx} — {analyzed}/{int(max_frames)} analysées")
                frame_idx += 1

            cap.release()
            os.unlink(vid_path)
            status.success(f"✅ {analyzed} frames analysées")

            if class_counts:
                st.subheader("📊 Distribution des classes")
                import pandas as pd
                df = pd.DataFrame(list(class_counts.items()), columns=["Classe","Nb"]).sort_values("Nb", ascending=False)
                st.bar_chart(df.set_index("Classe"))

            st.download_button("⬇️ Exporter JSON", json.dumps(all_results, indent=2),
                               file_name="classification_video.json", mime="application/json")
