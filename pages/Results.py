import io

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from common import cards, detect, draw_detections, header, setup, to_png_bytes

models, confidence, show_conf = setup("Results · Road Damage Detection")

image_bytes = st.session_state.get("image_bytes")
if image_bytes is None:
    header("Results", "There's no photo to analyse yet.")
    if st.button("Upload a road photo", type="primary"):
        st.switch_page("app.py")
    st.stop()

header(
    "Detection results",
    f"Photo: {st.session_state.get('image_name', 'uploaded image')}",
)

if st.button("Upload another photo", type="primary"):
    st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
    st.session_state.pop("image_bytes", None)
    st.switch_page("app.py")

image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")

with st.spinner("Analysing image…"):
    dets = detect(models, image, confidence)
annotated = draw_detections(image, dets, show_conf)

# ---------------------------------------------------------------- summary
st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
counts = {}
for d in dets:
    label = d["name"].capitalize()
    counts[label] = counts.get(label, 0) + 1
cards([("Total detections", len(dets))] + list(counts.items()))

if not dets:
    st.info(
        "No road damage was found at this confidence level. "
        "If you expect damage in the photo, lower the confidence threshold in the sidebar."
    )

# ---------------------------------------------------------------- images
st.markdown('<div class="section-title">Photos</div>', unsafe_allow_html=True)
left, right = st.columns(2, gap="large")
with left:
    st.image(image, caption="Original photo", use_container_width=True)
with right:
    st.image(annotated, caption="Detected damage", use_container_width=True)

st.download_button(
    "Download annotated image",
    data=to_png_bytes(annotated),
    file_name="road_damage_result.png",
    mime="image/png",
)

# ---------------------------------------------------------------- details
if dets:
    st.markdown('<div class="section-title">Detection details</div>', unsafe_allow_html=True)
    img_area = image.width * image.height
    rows = []
    for i, d in enumerate(dets, start=1):
        x1, y1, x2, y2 = d["box"]
        rows.append(
            {
                "#": i,
                "Damage type": d["name"].capitalize(),
                "Confidence": f"{d['conf']:.0%}",
                "Share of image": f"{((x2 - x1) * (y2 - y1) / img_area):.1%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.markdown(
    '<div class="footnote">Detections are automated estimates from YOLOv8 models trained on a '
    "public road damage dataset. Verify results on site before making any maintenance decision.</div>",
    unsafe_allow_html=True,
)
