import streamlit as st
from PIL import Image, ImageOps

from common import cards, header, setup  # noqa: F401

setup("Road Damage Detection")

header(
    "Road Damage Detection",
    "Upload a photo of a road surface. The model finds potholes and cracks, "
    "then shows the results on a separate page.",
)

upload_round = st.session_state.get("upload_round", 0)
uploaded = st.file_uploader(
    "Upload a road photo",
    type=["jpg", "jpeg", "png", "webp", "avif"],
    key=f"uploader_{upload_round}",
    help="Photos taken from above or from a vehicle dashboard give the best results.",
)

if uploaded is None:
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="cards">
          <div class="card" style="background:#F9E4EA"><div class="num">1</div>
            <div class="ttl">Upload a photo</div><div class="lab">JPG, PNG, WEBP or AVIF.</div></div>
          <div class="card" style="background:#E2F4EC"><div class="num">2</div>
            <div class="ttl">Automatic analysis</div><div class="lab">Two YOLOv8 models scan the road.</div></div>
          <div class="card" style="background:#E4EEFA"><div class="num">3</div>
            <div class="ttl">Review the results</div><div class="lab">Counts, boxes and a details table.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="footnote">Detections are automated estimates. '
        "Verify results on site before making any maintenance decision.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

try:
    Image.open(uploaded).verify()
except Exception:
    st.error(
        "This file could not be read as an image. "
        "Save it as a JPG or PNG and upload it again."
    )
    st.stop()

st.session_state["image_bytes"] = uploaded.getvalue()
st.session_state["image_name"] = uploaded.name
st.switch_page("pages/Results.py")
