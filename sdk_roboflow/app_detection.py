import streamlit as st
import tempfile
import os
import json
import io
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────
st.set_page_config(page_title="Détection App", page_icon="🎯", layout="wide")
st.title("🎯 Détection App — Roboflow")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key       = st.text_input("Clé API Roboflow", type="password")
    project_id    = st.text_input("Project ID", value="cv_lab_roboflow-9fbpq-vrod5")
    model_version = st.number_input("Version", min_value=1, value=1)
    confidence    = st.slider("Seuil de confiance (%)", 0, 100, 40)
    overlap       = st.slider("Seuil overlap (%)", 0, 100, 30)
    thickness     = st.slider("Épaisseur des boîtes", 1, 5, 2)

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
# PALETTE COULEURS PAR CLASSE
# ─────────────────────────────────────────────
COLORS = [
    (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 220, 80),
    (220, 80, 255), (80, 255, 220), (255, 160, 80), (80, 180, 255),
]
color_map = {}

def get_color(cls):
    if cls not in color_map:
        color_map[cls] = COLORS[len(color_map) % len(COLORS)]
    return color_map[cls]

# ─────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────
def predict(model, img_bytes, filename, conf, ovlp):
    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name
    try:
        result = model.predict(tmp_path, confidence=conf, overlap=ovlp).json()
        return result
    finally:
        os.unlink(tmp_path)

def parse_detections(result):
    preds = result.get("predictions", [])
    # Normaliser le nom de la classe
    for p in preds:
        if "class" not in p:
            p["class"] = p.get("class_name", "?")
    return preds

def draw_boxes(img_bytes, preds, thick=2):
    img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    for p in preds:
        cls  = p.get("class", "?")
        conf = p.get("confidence", 0) * 100
        x    = p.get("x", 0)
        y    = p.get("y", 0)
        w    = p.get("width", 0)
        h    = p.get("height", 0)

        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        color  = get_color(cls)

        # Boîte
        for t in range(thick):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color)

        # Label
        label = f"{cls} {conf:.0f}%"
        lw    = len(label) * 8 + 8
        draw.rectangle([x1, y1-22, x1+lw, y1], fill=color)
        draw.text((x1+4, y1-20), label, fill=(255, 255, 255))

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

            with st.spinner("Détection en cours..."):
                result = predict(st.session_state.model, img_bytes, uploaded.name, confidence, overlap)
                preds  = parse_detections(result)

            # ── Côte à côte ──
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🖼️ Image originale")
                st.image(img_bytes, use_column_width=True)
            with col2:
                st.subheader(f"🎯 Résultat — {len(preds)} objet(s)")
                ann = draw_boxes(img_bytes, preds, thickness)
                st.image(ann, use_column_width=True)

            # ── Liste des détections ──
            st.markdown("---")
            st.subheader(f"📦 {len(preds)} Détection(s)")

            if not preds:
                st.warning("Aucun objet détecté. Essaie de baisser le seuil de confiance dans la sidebar.")
                st.write("Réponse brute :", result)
            else:
                for i, p in enumerate(preds):
                    cls  = p.get("class", "?")
                    conf = p.get("confidence", 0) * 100
                    x, y, w, h = p.get("x",0), p.get("y",0), p.get("width",0), p.get("height",0)

                    col_a, col_b, col_c = st.columns([2, 1, 2])
                    col_a.progress(int(conf), text=f"**{cls}**")
                    col_b.metric("", f"{conf:.1f}%")
                    col_c.caption(f"x:{x:.0f} y:{y:.0f} | w:{w:.0f} h:{h:.0f}")

            # ── Export ──
            col_j, col_i = st.columns(2)
            col_j.download_button(
                "⬇️ Télécharger JSON",
                json.dumps({"file": uploaded.name, "detections": preds}, indent=2),
                file_name="detection.json", mime="application/json"
            )
            buf = io.BytesIO()
            draw_boxes(img_bytes, preds, thickness).save(buf, format="PNG")
            col_i.download_button(
                "⬇️ Image annotée PNG",
                buf.getvalue(),
                file_name="detection_annotee.png", mime="image/png"
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
        elif st.button("▶️ Lancer la détection", use_container_width=True):
            all_results  = []
            total_dets   = 0
            progress     = st.progress(0)
            cols         = st.columns(3)
            col_idx      = 0
            class_counts = {}

            for i, f in enumerate(files):
                try:
                    img_bytes = f.read()
                    result    = predict(st.session_state.model, img_bytes, f.name, confidence, overlap)
                    preds     = parse_detections(result)
                    ann       = draw_boxes(img_bytes, preds, thickness)
                    n         = len(preds)
                    total_dets += n
                    cols[col_idx].image(ann, caption=f"{f.name} — {n} obj.", use_column_width=True)
                    col_idx = (col_idx + 1) % 3
                    for p in preds:
                        cls = p.get("class","?")
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                    all_results.append({"file": f.name, "detections": preds})
                except Exception as e:
                    all_results.append({"file": f.name, "error": str(e)})
                progress.progress((i+1)/len(files))

            st.success(f"✅ {len(files)} images — {total_dets} objet(s) détecté(s) au total")

            if class_counts:
                st.subheader("📊 Distribution des classes")
                import pandas as pd
                df = pd.DataFrame(list(class_counts.items()), columns=["Classe","Nb"]).sort_values("Nb", ascending=False)
                st.bar_chart(df.set_index("Classe"))

            st.download_button("⬇️ Exporter JSON", json.dumps(all_results, indent=2),
                               file_name="detection_dossier.json", mime="application/json")

# ══════════════════
# TAB 3 — VIDÉO
# ══════════════════
with tab3:
    video_file   = st.file_uploader("Choisir une vidéo", type=["mp4","avi","mov","mkv"])
    col_a, col_b = st.columns(2)
    frame_skip   = col_a.number_input("Analyser 1 frame sur N", min_value=1, value=3)
    max_frames   = col_b.number_input("Max frames", min_value=1, value=50)
    export_video = st.checkbox("Exporter la vidéo annotée (MP4)", value=False)

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
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_idx    = 0
            analyzed     = 0
            all_results  = []
            class_counts = {}
            progress     = st.progress(0)
            status       = st.empty()
            preview      = st.empty()

            out_path = None
            writer   = None
            if export_video:
                out_path = vid_path.replace(".mp4", "_annote.mp4")
                writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            while cap.isOpened() and analyzed < int(max_frames):
                ret, frame = cap.read()
                if not ret: break

                if frame_idx % int(frame_skip) == 0:
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    buf     = io.BytesIO()
                    img_pil.save(buf, format="JPEG")
                    img_bytes = buf.getvalue()

                    result = predict(st.session_state.model, img_bytes, "frame.jpg", confidence, overlap)
                    preds  = parse_detections(result)
                    ann    = draw_boxes(img_bytes, preds, thickness)

                    preview.image(ann, caption=f"Frame {frame_idx} — {len(preds)} obj. — t={frame_idx/fps:.1f}s", use_column_width=True)

                    if writer:
                        bgr = cv2.cvtColor(np.array(ann), cv2.COLOR_RGB2BGR)
                        writer.write(bgr)

                    for p in preds:
                        cls = p.get("class","?")
                        class_counts[cls] = class_counts.get(cls, 0) + 1

                    all_results.append({
                        "frame": frame_idx,
                        "time_s": round(frame_idx/fps, 2),
                        "detections": preds
                    })
                    analyzed += 1
                    progress.progress(min(analyzed/int(max_frames), 1.0))
                    status.text(f"Frame {frame_idx} — {analyzed}/{int(max_frames)} analysées")

                frame_idx += 1

            cap.release()
            if writer: writer.release()
            os.unlink(vid_path)

            status.success(f"✅ {analyzed} frames analysées — {sum(class_counts.values())} détections totales")

            # Résumé
            c1, c2, c3 = st.columns(3)
            c1.metric("Frames analysées", analyzed)
            c2.metric("Détections totales", sum(class_counts.values()))
            c3.metric("Classes distinctes", len(class_counts))

            if class_counts:
                st.subheader("📊 Distribution des classes")
                import pandas as pd
                df = pd.DataFrame(list(class_counts.items()), columns=["Classe","Nb"]).sort_values("Nb", ascending=False)
                st.bar_chart(df.set_index("Classe"))

            st.download_button("⬇️ Exporter JSON", json.dumps(all_results, indent=2),
                               file_name="detection_video.json", mime="application/json")

            if export_video and out_path and os.path.exists(out_path):
                with open(out_path, "rb") as vf:
                    st.download_button("⬇️ Télécharger vidéo annotée", vf.read(),
                                       file_name="detection_annotee.mp4", mime="video/mp4")
                os.unlink(out_path)
