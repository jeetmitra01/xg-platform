import requests
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mplsoccer import Pitch

@st.cache_resource
def get_pitch():
    # Using a green pitch with white lines to match the requested aesthetic
    return Pitch(
        pitch_type="statsbomb", 
        pitch_color='#2e7d32', 
        line_color='white', 
        stripe=True, 
        stripe_color='#2b732e'
    )

def render_shot_preview(x: float, y: float, label: str = None):
    pitch = get_pitch()
    fig, ax = pitch.draw(figsize=(7, 5))
    if label is None:
        label = f"({x:.1f}, {y:.1f})"
    # Position text below the point
    txt = ax.text(x, y + 4, label, fontsize=10, ha='center', va='top', fontweight='bold', color='white')
    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
    pitch.scatter([x], [y], s=120, ax=ax, edgecolors='white', color='#0078ff', zorder=3)
    ax.set_title("Shot location preview", fontweight='bold', fontsize=14, pad=15)
    return fig


API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="xG Predictor", layout="centered")
st.title("⚽ Real-Time xG Predictor")
st.caption("Calls a FastAPI inference service and returns predicted expected goals (xG).")
st.caption("For more info on the model check out the [la-liga-xg](https://github.com/jeetmitra01/la-liga-xg) repo.")
st.caption("The Goal is on the right side of the pitch.")
# Initialize session state for prediction tracking
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

col1, col2 = st.columns(2)
with col1:
    x = st.slider("Shot X (0–120)", 0.0, 120.0, 102.4, 0.1)
with col2:
    y = st.slider("Shot Y (0–80)", 0.0, 80.0, 41.2, 0.1)

shot_body_part = st.selectbox("Body part", ["Left Foot", "Right Foot", "Head", "Other"])
shot_type = st.selectbox("Shot type", ["Open Play", "Free Kick", "Corner", "Throw-in", "Other"])
play_pattern = st.selectbox("Play pattern", ["Regular Play", "From Corner", "From Free Kick", "From Throw In", "Other"])

# Check if the current position matches the last prediction
display_label = None
if st.session_state.last_prediction:
    lp = st.session_state.last_prediction
    if lp["x"] == x and lp["y"] == y:
        display_label = f"xG: {lp['xg']:.3f}"

st.subheader("Shot Preview")
fig = render_shot_preview(x, y, label=display_label)
st.pyplot(fig, clear_figure=True)

if st.session_state.last_prediction and display_label:
    lp = st.session_state.last_prediction
    st.metric("Predicted xG", f"{lp['xg']:.3f}")
    st.caption(f"Model: {lp['model_version']} • Latency: {lp['latency_ms']:.1f} ms")

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

    st.session_state.last_prediction = {
        "x": x,
        "y": y,
        "xg": out['xg'],
        "model_version": out['model_version'],
        "latency_ms": out['latency_ms']
    }
    st.rerun()
