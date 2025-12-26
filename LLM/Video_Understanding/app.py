import streamlit as st
import subprocess
import sys
import os
import time
import pandas as pd

st.set_page_config(
    page_title="Video → Text Understanding Pipeline",
    layout="wide"
)

st.title("Video → Text Understanding Flow I")

# ---------- Helpers ----------

def run_script(command, status_label, log_container):
    """
    Runs a python script as a subprocess and streams stdout to Streamlit
    """
    with st.status(status_label, expanded=True) as status:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            log_container.write(line)

        process.wait()

        if process.returncode == 0:
            status.update(label=f"✅ {status_label}", state="complete")
        else:
            status.update(label=f"❌ {status_label}", state="error")
            st.stop()


# ---------- UI Inputs ----------

st.subheader("Input")

video_url = st.text_input(
    "Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

run_clicked = st.button("▶ Run Pipeline", type="primary")

# ---------- Execution ----------

if run_clicked:
    if not video_url:
        st.error("Please enter a video URL")
        st.stop()

    # Clean old outputs (optional but recommended)
    for f in ["input.mp4", "processed_video.mp4", "outputs.csv", "summary.txt"]:
        if os.path.exists(f):
            os.remove(f)

    if os.path.exists("video_chunks"):
        for f in os.listdir("video_chunks"):
            os.remove(os.path.join("video_chunks", f))

    st.divider()
    st.subheader("Pipeline Progress")

    log_box = st.container()

    # ---------------- STEP 1 ----------------
    run_script(
        [
            sys.executable,
            "video_downloader.py",
            video_url
        ],
        "Downloading video",
        log_box
    )

    # ---------------- STEP 2 ----------------
    run_script(
        [
            sys.executable,
            "video_preprocess.py"
        ],
        "Preprocessing & chunking video",
        log_box
    )

    # ---------------- STEP 3 ----------------
    run_script(
        [
            sys.executable,
            "orchestrator.py"
        ],
        "Running video chunk inference (VLM)",
        log_box
    )

    # ---------------- STEP 4 ----------------
    run_script(
        [
            sys.executable,
            "reconcile.py"
        ],
        "Reconciling responses & generating summary",
        log_box
    )

    # ---------- RESULTS ----------

    st.divider()
    st.subheader("Results")

    col1, col2 = st.columns(2)

    # Summary
    with col1:
        st.markdown("### 📄 Final Summary")

        if os.path.exists("summary.txt"):
            with open("summary.txt", "r", encoding="utf-8") as f:
                summary_text = f.read()
            st.text_area(
                "Summary",
                summary_text,
                height=300
            )

            st.download_button(
                "⬇ Download summary.txt",
                summary_text,
                file_name="summary.txt"
            )
        else:
            st.warning("summary.txt not found")

    # CSV
    with col2:
        st.markdown("### 📊 Chunk-wise Descriptions")

        if os.path.exists("outputs.csv"):
            df = pd.read_csv("outputs.csv")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇ Download outputs.csv",
                df.to_csv(index=False),
                file_name="outputs.csv",
                mime="text/csv"
            )
        else:
            st.warning("outputs.csv not found")

    st.success("🎉 Pipeline completed successfully")
