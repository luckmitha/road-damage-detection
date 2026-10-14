import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Lets Streamlit open AVIF images when the plugin is installed.
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

# ---------------------------------------------------------------- settings
MODEL_PATHS = ["road_best.pt", "pothole_best.pt"]  # your trained YOLOv8 weights
DEFAULT_CONFIDENCE = 0.50
INK = "#3B3650"

CLASS_COLORS = {
    "pothole": "#EE7C95",
    "alligator cracking": "#5CBF98",
    "transverse cracking": "#7FA9E8",
    "longitudinal cracking": "#F2B36B",
    "edge cracking": "#A98BE0",
    "patching": "#E58BC8",
    "rutting": "#5BBFC4",
}
FALLBACK_COLORS = ["#EE7C95", "#5CBF98", "#7FA9E8", "#F2B36B", "#A98BE0", "#5BBFC4"]
CARD_TINTS = ["#F9E4EA", "#E2F4EC", "#E4EEFA", "#FFF0E3", "#EEE8FA"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, button, input { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none; }

.block-container { max-width: 1120px; padding-top: 5rem; padding-bottom: 3rem; }

.app-header { border-left: 6px solid #B9A7E8; padding: 4px 0 4px 18px; margin-bottom: 1.8rem; }
.app-title { font-family: 'Poppins', sans-serif; font-size: 2.7rem; font-weight: 600;
             letter-spacing: -0.02em; color: #3B3650; margin: 0; line-height: 1.25; }
.app-sub { color: #7B7590; font-size: 1.05rem; margin: 0.5rem 0 0; max-width: 62ch; }

.section-title { font-family: 'Poppins', sans-serif; font-size: 1.15rem; font-weight: 500;
                 color: #3B3650; margin: 2rem 0 0.7rem; }

[data-testid="stFileUploaderDropzone"] { background: #FFFFFF; border: 1.5px dashed #CDBDEB;
                                          border-radius: 12px; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; border: 1px solid #D9CCF0;
                                                  background: #FFFFFF; color: #3B3650; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #9B86DB; color: #6B54B8; }
.stButton > button[kind="primary"] { background: #9B86DB; border-color: #9B86DB; color: #FFFFFF; }
.stButton > button[kind="primary"]:hover { background: #8A74CF; color: #FFFFFF; }

.cards { display: flex; gap: 14px; flex-wrap: wrap; }
.card { flex: 1; min-width: 150px; border-radius: 12px; padding: 14px 18px; border: 1px solid #E6DCF0; }
.card .lab { color: #5F5A75; font-size: 0.88rem; }
.card .num { font-family: 'Poppins', sans-serif; font-size: 2rem; font-weight: 500; color: #3B3650; }
.card .ttl { font-family: 'Poppins', sans-serif; font-weight: 500; color: #3B3650; margin-top: 2px; }

.legend-row { display: flex; align-items: center; gap: 10px; margin: 6px 0;
              font-size: 0.92rem; color: #3B3650; }
.legend-dot { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }

.footnote { color: #8E88A3; font-size: 0.82rem; margin-top: 2.5rem;
            border-top: 1px solid #E6DCF0; padding-top: 0.9rem; }
</style>
"""


# ---------------------------------------------------------------- colours
def color_for(name: str) -> str:
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    return FALLBACK_COLORS[sum(map(ord, name)) % len(FALLBACK_COLORS)]


# ---------------------------------------------------------------- page setup
@st.cache_resource(show_spinner="Loading models…")
def load_models(paths: tuple):
    return [YOLO(p) for p in paths]


def setup(page_title: str):
    """Common page setup: theme, models and the settings sidebar."""
    st.set_page_config(page_title=page_title, page_icon="🛣️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    missing = [p for p in MODEL_PATHS if not Path(p).exists()]
    if missing:
        st.error(
            f"Model file not found: **{', '.join(missing)}**. "
            "Place the weights next to app.py or update MODEL_PATHS in common.py."
        )
        st.stop()
    models = load_models(tuple(MODEL_PATHS))
    all_names = sorted({n for m in models for n in m.names.values()})

    with st.sidebar:
        st.markdown("### Detection settings")
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=st.session_state.get("conf", DEFAULT_CONFIDENCE),
            step=0.05,
            key="conf_slider",
            help=(
                "Minimum certainty required to show a detection. "
                "Raise it to remove false alarms, lower it to catch more damage."
            ),
        )
        st.session_state["conf"] = confidence
        show_conf = st.toggle(
            "Show confidence on labels",
            value=st.session_state.get("show_conf", True),
            key="show_conf_toggle",
        )
        st.session_state["show_conf"] = show_conf

        st.markdown("### Damage types")
        legend = "".join(
            f'<div class="legend-row"><span class="legend-dot" style="background:{color_for(n)}"></span>'
            f"{n.capitalize()}</div>"
            for n in all_names
        )
        st.markdown(legend, unsafe_allow_html=True)

    return models, confidence, show_conf


def header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="app-header">
            <p class="app-title">{title}</p>
            <p class="app-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cards(items):
    """Pastel summary cards. items = [(label, value), ...]"""
    html = "".join(
        f'<div class="card" style="background:{CARD_TINTS[i % len(CARD_TINTS)]}">'
        f'<div class="lab">{label}</div><div class="num">{value}</div></div>'
        for i, (label, value) in enumerate(items)
    )
    st.markdown(f'<div class="cards">{html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- detection
def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def detect(models, image: Image.Image, conf: float):
    """Run every model, then drop duplicate boxes of the same damage type."""
    found = []
    for m in models:
        r = m.predict(image, conf=conf, verbose=False)[0]
        for b in r.boxes:
            found.append(
                {
                    "name": m.names[int(b.cls[0])],
                    "conf": float(b.conf[0]),
                    "box": [float(v) for v in b.xyxy[0]],
                }
            )
    found.sort(key=lambda d: d["conf"], reverse=True)
    kept = []
    for d in found:
        if not any(k["name"] == d["name"] and iou(k["box"], d["box"]) > 0.5 for k in kept):
            kept.append(d)
    return kept


def get_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_detections(image: Image.Image, dets, show_conf: bool) -> Image.Image:
    """Draw boxes with small pastel label tags on a copy of the image."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    line = max(3, round(min(w, h) / 200))
    font = get_font(max(13, round(min(w, h) / 38)))

    for d in dets:
        x1, y1, x2, y2 = d["box"]
        color = color_for(d["name"])
        label = d["name"].capitalize()
        if show_conf:
            label += f" {d['conf']:.0%}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line)

        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        pad = 4
        tag_top = max(0, y1 - th - 2 * pad)
        draw.rectangle([x1, tag_top, x1 + tw + 2 * pad, tag_top + th + 2 * pad], fill=color)
        draw.text((x1 + pad, tag_top + pad - top), label, fill=INK, font=font)
    return out


def to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
