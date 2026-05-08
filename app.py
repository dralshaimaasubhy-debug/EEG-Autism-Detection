import sys
import subprocess

def ensure_package(pkg):
    try:
        __import__(pkg)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Ensure required packages
for p in ["numpy", "pandas", "matplotlib", "seaborn", "reportlab"]:
    ensure_package(p)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
import time
from datetime import datetime
import os

# ================== SESSION STATE ==================
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "results" not in st.session_state:
    st.session_state.results = {}

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="EEG Autism Detection", layout="wide")

st.markdown("""
<style>
h1 {color:#800020; text-align:center;}
h2, h3 {color:#800020;}
.section {background-color:#f7f7f7; padding:20px; border-radius:12px; margin-bottom:20px;}
.box-normal {background-color:#eaf7ea; padding:20px; border-left:6px solid #2e7d32; border-radius:12px;}
.box-autism {background-color:#fdecea; padding:20px; border-left:6px solid #c62828; border-radius:12px;}
.box-confidence {background-color:#e3f2fd; padding:20px; border-left:6px solid #1565c0; border-radius:12px;}
.box-explain {background-color:#fff8e1; padding:20px; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 EEG Autism Detection System")
st.markdown("### ALDaayen School for Girls")

# ================== LOAD MODEL ==================
with open("autism_model.pkl", "rb") as f:
    model = pickle.load(f)
FEATURE_COLS = [f"ch{i}" for i in range(1, 17)]
HELMET_TO_CH = {
    "FP1":"ch1","FP2":"ch2","F3":"ch3","F4":"ch4","FZ":"ch5","CZ":"ch6",
    "T7":"ch7","T8":"ch8","C3":"ch9","C4":"ch10",
    "P7":"ch11","P8":"ch12","P3":"ch13","P4":"ch14","O1":"ch15","O2":"ch16"
}

def prepare_features(df):
    if "FP1" in df.columns:
        df = df.rename(columns=HELMET_TO_CH)
    return df[FEATURE_COLS]

# ================== PATIENT INFO ==================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("👤 Patient Information")
name = st.text_input("Patient Name")
age = st.number_input("Age (months)", min_value=1, max_value=60)
st.markdown("</div>", unsafe_allow_html=True)

# ================== UPLOAD ==================
st.markdown("<div class='section'>", unsafe_allow_html=True)
file = st.file_uploader("📁 Upload EEG CSV (Helmet or Training)")
st.markdown("</div>", unsafe_allow_html=True)

# ================== ANALYZE ==================
if file:
    if st.button("🔍 Analyze EEG File"):
        with st.spinner("⏳ Analyzing EEG data..."):
            time.sleep(1)

            try:
                df_raw = pd.read_csv(file, encoding="utf-8")
            except:
                try:
                    df_raw = pd.read_csv(file, encoding="latin1")
                except:
                    df_raw = pd.read_csv(file, encoding="cp1256")

            data = prepare_features(df_raw)

            # Feature extraction
            alpha_like = data.mean(axis=1)
            theta_like = data.std(axis=1)
            gamma_like = data.max(axis=1)

            channel_means = data.mean()

            pred = model.predict(data)
            prob = model.predict_proba(data)

            diagnosis = "Autism" if pred[0] == 1 else "Normal"
            confidence = round(float(np.max(prob[0]) * 100), 2)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

            temp_dir = tempfile.mkdtemp()

            # Channel Amplitude Plot
            fig1, ax1 = plt.subplots(figsize=(8,4))
            ax1.bar(channel_means.index, channel_means.values, color="#1565c0")
            ax1.set_title("EEG Channel Amplitude Distribution")
            ax1.set_xlabel("EEG Channels")
            ax1.set_ylabel("Amplitude")
            plt.xticks(rotation=45)
            amp_path = os.path.join(temp_dir, "amplitude.png")
            fig1.savefig(amp_path)
            plt.close(fig1)

            # Synchronization Heatmap
            fig2, ax2 = plt.subplots()
            sns.heatmap(data.corr(), cmap="coolwarm", ax=ax2)
            ax2.set_title("Inter-channel Synchronization")
            sync_path = os.path.join(temp_dir, "sync.png")
            fig2.savefig(sync_path)
            plt.close(fig2)

            st.session_state.results = {
                "diagnosis": diagnosis,
                "confidence": confidence,
                "time": timestamp,
                "amp": amp_path,
                "sync": sync_path
            }

            st.session_state.analyzed = True

        st.success("✅ File analyzed successfully")

# ================== SHOW RESULTS ==================
if st.session_state.analyzed:
    r = st.session_state.results

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='{'box-autism' if r['diagnosis']=='Autism' else 'box-normal'}'><h2>{r['diagnosis']}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='box-confidence'><h2>{r['confidence']}%</h2></div>", unsafe_allow_html=True)

    st.subheader("📊 EEG Channel Amplitude Profile")
    st.image(r["amp"])

    st.subheader("🔗 Inter-channel Synchronization")
    st.image(r["sync"])

    st.markdown("<div class='box-explain'>", unsafe_allow_html=True)
    st.subheader("🧠 Diagnosis based on EEG features:")

    if r["diagnosis"] == "Autism":
        st.write("- Increased Theta-like variability")
        st.write("- Irregular Gamma-related activity")
        st.write("- Disrupted inter-channel synchronization")
        st.write("- Abnormal Alpha-like activity pattern")
    else:
        st.write("- Balanced Alpha-like activity")
        st.write("- Low Theta-like variability")
        st.write("- Normal inter-channel synchronization")
        st.write("- Stable Alpha-like activity")

    st.markdown("</div>", unsafe_allow_html=True)

    # ================== PDF REPORT ==================
    def generate_pdf():
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        c = canvas.Canvas(tmp.name, pagesize=A4)

        if os.path.exists("school_logo.png"):
            c.drawImage("school_logo.png", 50, 770, 60, 60)

        c.setFont("Helvetica-Bold", 16)
        c.drawString(130, 800, "EEG Autism Detection Report")
        c.setFont("Helvetica", 11)
        c.drawString(130, 780, f"Date & Time: {r['time']}")

        c.line(50, 760, 550, 760)

        c.drawString(50, 730, f"Patient Name: {name}")
        c.drawString(50, 710, f"Age (months): {age}")
        c.drawString(50, 690, f"Diagnosis: {r['diagnosis']}")
        c.drawString(50, 670, f"Confidence: {r['confidence']}%")

        c.drawImage(r["amp"], 50, 430, width=500, height=150)
        c.drawImage(r["sync"], 50, 250, width=500, height=150)

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 230, "Diagnosis based on EEG features:")
        c.setFont("Helvetica", 11)
        y = 210

        if r["diagnosis"] == "Autism":
            points = [
                "Increased Theta-like variability",
                "Irregular Gamma-related activity",
                "Disrupted inter-channel synchronization",
                "Abnormal Alpha-like activity pattern"
            ]
        else:
            points = [
                "Balanced Alpha-like activity",
                "Low Theta-like variability",
                "Normal inter-channel synchronization",
                "Stable Alpha-like activity"
            ]

        for p in points:
            c.drawString(60, y, f"- {p}")
            y -= 18

        c.setFont("Helvetica-Oblique", 9)
        c.drawString(50, 100, "Generated by EEG Autism Detection System (AI-based)")

        c.save()
        return tmp.name

    pdf_path = generate_pdf()
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Medical Report (PDF)", f, file_name="EEG_Medical_Report.pdf")
