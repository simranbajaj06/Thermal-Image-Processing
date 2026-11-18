"""
WORKING DEMONSTRATION (the 'final demonstration' part of the assignment)
Run:  streamlit run app.py
Choose a source (built-in synthetic demo OR a folder of images), press the button, see the results.
"""
import os
import numpy as np
import streamlit as st

from src import io_utils, pipeline
from src.synth import make_synthetic

st.set_page_config(page_title="Thermal perfusion-pattern analysis", layout="wide")
st.title("Thermal image sequence -> perfusion-related pattern")
st.caption("Surface temperature only: this shows WHERE the skin warms/cools and WHICH WAY the change spreads. "
           "It does not measure blood flow directly and is not a diagnosis.")

with st.sidebar:
    st.header("Input")
    source = st.radio("Source", ["Synthetic demo (known answer)", "Folder of images"])
    if source.startswith("Synthetic"):
        true_angle = st.slider("True direction of the simulated warming front (deg)", 0, 359, 135)
        noise = st.slider("Sensor noise", 0.0, 0.05, 0.01, 0.005)
        fps = st.number_input("Frames per second", 1.0, 60.0, 5.0)
        folder, scale, stride = None, 1.0, 1
    else:
        folder = st.text_input("Folder with the frames (time-ordered images)", "data/DIRT-SR38/...")
        fps = st.number_input("Frames per second of the recording", 0.1, 60.0, 9.0)
        scale = st.select_slider("Resize factor (smaller = faster)", [0.25, 0.5, 0.75, 1.0], 0.5)
        stride = st.number_input("Use every n-th frame", 1, 50, 1)
        true_angle, noise = None, None
    roi_mode = st.radio("Region of interest", ["auto", "full"], index=0, horizontal=True,
                        help="auto = find a body on a background; full = whole frame is skin (close-up recordings)")
    register = st.checkbox("Motion correction (slower)", False)
    run_btn = st.button("Run analysis", type="primary")

if run_btn:
    with st.spinner("Analysing ... (the video takes the longest)"):
        if folder is None:
            T, mask, info = make_synthetic(true_angle, noise=noise, fps=fps)
            name, f_eff = f"synthetic_{true_angle}deg", fps
        else:
            T, _ = io_utils.load_sequence(folder, int(stride), None, scale)
            name, f_eff = os.path.basename(os.path.normpath(folder)), fps / int(stride)
        out = os.path.join("results", "app_run")
        res = pipeline.run(T, f_eff, out, name, cfg=pipeline.Config(register=register, roi_mode=roi_mode),
                           video_stride=max(1, len(T) // 80))
    st.session_state["out"] = out
    st.session_state["res"] = res
    st.session_state["true"] = true_angle

if "res" in st.session_state:
    res, out = st.session_state["res"], st.session_state["out"]
    st.subheader("Summary")
    st.text(res["summary_text"])
    pr = res["phase_results"].get("rewarming")
    if pr and st.session_state.get("true") is not None:
        from src.flow import angular_diff
        err = float(angular_diff(pr["statsB"]["mean_deg"], st.session_state["true"]))
        st.success(f"True direction {st.session_state['true']} deg  |  estimated {pr['statsB']['mean_deg']:.1f} deg  "
                   f"|  error {err:.1f} deg")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Frame-by-frame video")
        vid = os.path.join(out, "demo.mp4")
        if os.path.exists(vid):
            st.video(vid)
    with c2:
        st.subheader("Increasing vs decreasing regions over time")
        st.image(os.path.join(out, "rising_falling_over_time.png"))
    st.subheader("Overview figure")
    st.image(os.path.join(out, "summary.png"))
    st.subheader("Frame-by-frame numbers")
    st.dataframe(res["table"], height=250)
else:
    st.info("Pick a source in the sidebar and press **Run analysis**.")
