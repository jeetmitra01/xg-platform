import requests
import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch

@st.cache_resource
def get_pitch():
    return Pitch(pitch_type="statsbomb")

def render_shot_preview(x: float, y: float):
    pitch = get_pitch()
    fig, ax = pitch.draw(figsize=(7, 5))
    ax.text(x + 1, y + 1, f"({x:.1f}, {y:.1f})", fontsize=10)
    pitch.scatter([x], [y], s=250, ax=ax)
    ax.set_title("Shot location preview")
    return fig


API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="xG Predictor", layout="centered")
st.title("⚽ Real-Time xG Predictor")
st.caption("Calls a FastAPI inference service and returns predicted expected goals (xG).")

col1, col2 = st.columns(2)
with col1:
    x = st.slider("Shot X (0–120)", 0.0, 120.0, 102.4, 0.1)
with col2:
    y = st.slider("Shot Y (0–80)", 0.0, 80.0, 41.2, 0.1)

st.subheader("Shot Preview")
fig = render_shot_preview(x, y)
st.pyplot(fig, clear_figure=True)


shot_body_part = st.selectbox("Body part", ["Left Foot", "Right Foot", "Head", "Other"])
shot_type = st.selectbox("Shot type", ["Open Play", "Free Kick", "Corner", "Throw-in", "Other"])
play_pattern = st.selectbox("Play pattern", ["Regular Play", "From Corner", "From Free Kick", "From Throw In", "Other"])

if st.button("Predict xG"):
    payload = {
        "x": x,
        "y": y,
        "shot_body_part": shot_body_part,
        "shot_type": shot_type,
        "play_pattern": play_pattern,
    }
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    out = r.json()

    st.metric("Predicted xG", f"{out['xg']:.3f}")
    st.caption(f"Model: {out['model_version']} • Latency: {out['latency_ms']:.1f} ms")

    pitch = Pitch(pitch_type="statsbomb")
    fig, ax = pitch.draw(figsize=(7, 5))
    pitch.scatter([x], [y], s=250, ax=ax)
    ax.set_title("Shot location")
    st.pyplot(fig)
