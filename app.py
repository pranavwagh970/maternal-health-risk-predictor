# =============================================================
#  Maternal Health Risk Predictor - Streamlit Web App
#  DSBDAL Mini Project | TE IT B | AY 2025-26
#  Run with: py -3 -m streamlit run app.py
# =============================================================

from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]
MODEL_FILES = {
    "model": BASE_DIR / "model.pkl",
    "scaler": BASE_DIR / "scaler.pkl",
    "label_encoder": BASE_DIR / "label_encoder.pkl",
}

st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    missing_files = [path.name for path in MODEL_FILES.values() if not path.exists()]
    if missing_files:
        missing_list = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Missing model files: {missing_list}. Run `py -3 train_model.py` in this folder first."
        )

    model = joblib.load(MODEL_FILES["model"])
    scaler = joblib.load(MODEL_FILES["scaler"])
    label_encoder = joblib.load(MODEL_FILES["label_encoder"])
    return model, scaler, label_encoder


try:
    model, scaler, le = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


with st.sidebar:
    st.markdown("## About This App")
    st.markdown(
        "This tool predicts maternal health risk level during pregnancy "
        "using 6 key vitals collected from monitoring data."
    )
    st.markdown("---")
    st.markdown("**Dataset:** Maternal Health Risk")
    st.markdown("**Rows:** 1,014 patient records")
    st.markdown("**Model:** Best model selected during training")
    st.markdown("**Accuracy:** Usually around 85% to 90%, depending on the split")
    st.markdown("---")
    st.markdown("**Risk Levels**")
    st.success("Low Risk - Continue regular checkups")
    st.warning("Mid Risk - Increased monitoring needed")
    st.error("High Risk - Immediate medical attention")
    st.markdown("---")
    st.markdown("**Project:** DSBDAL Mini Project")
    st.markdown("**Class:** TE IT B | AY 2025-26")


st.title("Maternal Health Risk Predictor")
st.markdown(
    "Enter the patient's vital readings below to predict the **maternal health risk level** "
    "during pregnancy. All values should come from recent clinical measurements."
)
st.markdown("---")

st.subheader("Patient Vitals Input")
col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age (years)",
        min_value=10,
        max_value=70,
        value=25,
        help="Age of the pregnant woman in years",
    )
    systolic_bp = st.slider(
        "Systolic Blood Pressure (mmHg)",
        min_value=70,
        max_value=160,
        value=120,
        help="Upper number of the blood pressure reading",
    )
    diastolic_bp = st.slider(
        "Diastolic Blood Pressure (mmHg)",
        min_value=49,
        max_value=100,
        value=80,
        help="Lower number of the blood pressure reading",
    )

with col2:
    bs = st.number_input(
        "Blood Glucose Level (mmol/L)",
        min_value=6.0,
        max_value=19.0,
        value=7.0,
        step=0.1,
        help="Fasting blood glucose concentration in mmol/L",
    )
    body_temp = st.number_input(
        "Body Temperature (F)",
        min_value=98.0,
        max_value=104.0,
        value=98.6,
        step=0.1,
        help="Current body temperature in Fahrenheit",
    )
    heart_rate = st.slider(
        "Heart Rate (bpm)",
        min_value=60,
        max_value=90,
        value=72,
        help="Resting heart rate in beats per minute",
    )


with st.expander("View Normal Ranges Reference"):
    ref_df = pd.DataFrame(
        {
            "Feature": [
                "Age",
                "Systolic BP",
                "Diastolic BP",
                "Blood Glucose",
                "Body Temp",
                "Heart Rate",
            ],
            "Normal Range": [
                "18 to 35 years",
                "90 to 120 mmHg",
                "60 to 80 mmHg",
                "6.0 to 7.5 mmol/L",
                "97 to 99 F",
                "60 to 80 bpm",
            ],
            "Your Input": [
                f"{age} years",
                f"{systolic_bp} mmHg",
                f"{diastolic_bp} mmHg",
                f"{bs:.1f} mmol/L",
                f"{body_temp:.1f} F",
                f"{heart_rate} bpm",
            ],
        }
    )
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

st.markdown("---")
predict_clicked = st.button("Predict Risk Level", type="primary", use_container_width=True)

if predict_clicked:
    input_df = pd.DataFrame(
        [[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
        columns=FEATURE_COLUMNS,
    )
    input_scaled = scaler.transform(input_df)

    prediction = int(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0]
    class_ids = list(model.classes_)
    class_labels = le.inverse_transform(class_ids)
    predicted_class_index = class_ids.index(prediction)
    risk_label = le.inverse_transform([prediction])[0]
    confidence = probabilities[predicted_class_index] * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    if "high" in risk_label.lower():
        st.error(f"**{risk_label.upper()}** - {confidence:.1f}% confidence")
        st.markdown(
            "> **Recommendation:** Immediate medical consultation is strongly advised. "
            "Elevated readings can indicate possible complications."
        )
    elif "mid" in risk_label.lower():
        st.warning(f"**{risk_label.upper()}** - {confidence:.1f}% confidence")
        st.markdown(
            "> **Recommendation:** Increased monitoring is recommended. "
            "Schedule more frequent prenatal visits and track vitals closely."
        )
    else:
        st.success(f"**{risk_label.upper()}** - {confidence:.1f}% confidence")
        st.markdown(
            "> **Recommendation:** Continue with regular prenatal checkups and maintain healthy routines."
        )

    st.markdown("#### Prediction Confidence Breakdown")
    color_map = {
        "low risk": "#5DCAA5",
        "mid risk": "#EF9F27",
        "high risk": "#E24B4A",
    }
    bar_colors = [color_map.get(label.lower(), "#888780") for label in class_labels]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(class_labels, probabilities * 100, color=bar_colors, height=0.45, edgecolor="white")
    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + (bar.get_height() / 2),
            f"{prob * 100:.1f}%",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Model Confidence by Risk Class", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Patient Vitals Summary")
    vitals = {
        "Age": (age, 10, 70, 18, 35),
        "Systolic BP": (systolic_bp, 70, 160, 90, 120),
        "Diastolic BP": (diastolic_bp, 49, 100, 60, 80),
        "Blood Glucose": (bs, 6.0, 19.0, 6.0, 7.5),
        "Body Temp": (body_temp, 98.0, 104.0, 97.0, 99.0),
        "Heart Rate": (heart_rate, 60, 90, 60, 80),
    }

    fig2, axes = plt.subplots(2, 3, figsize=(12, 5))
    for ax2, (name, (val, vmin, vmax, nmin, nmax)) in zip(axes.flatten(), vitals.items()):
        ax2.barh([0], [vmax - vmin], left=vmin, color="#F1EFE8", height=0.5)
        ax2.barh([0], [nmax - nmin], left=nmin, color="#9FE1CB", height=0.5, alpha=0.7)
        marker_color = "#E24B4A" if (val < nmin or val > nmax) else "#5DCAA5"
        ax2.scatter([val], [0], color=marker_color, s=120, zorder=5)
        ax2.set_xlim(vmin, vmax)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_title(f"{name}\n{val}", fontsize=10, fontweight="bold")
        ax2.axis("off")

    normal_patch = mpatches.Patch(color="#9FE1CB", label="Normal range")
    green_patch = mpatches.Patch(color="#5DCAA5", label="Within normal")
    red_patch = mpatches.Patch(color="#E24B4A", label="Outside normal")
    fig2.legend(
        handles=[normal_patch, green_patch, red_patch],
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    fig2.suptitle("Vital Signs vs Normal Range", fontsize=13, fontweight="bold")
    fig2.patch.set_alpha(0)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    st.pyplot(fig2)

    st.markdown("#### Input Summary")
    summary_df = pd.DataFrame(
        {
            "Parameter": [
                "Age",
                "Systolic BP",
                "Diastolic BP",
                "Blood Glucose",
                "Body Temp",
                "Heart Rate",
            ],
            "Value": [
                f"{age} years",
                f"{systolic_bp} mmHg",
                f"{diastolic_bp} mmHg",
                f"{bs:.1f} mmol/L",
                f"{body_temp:.1f} F",
                f"{heart_rate} bpm",
            ],
            "Normal Range": [
                "18 to 35 years",
                "90 to 120 mmHg",
                "60 to 80 mmHg",
                "6.0 to 7.5 mmol/L",
                "97 to 99 F",
                "60 to 80 bpm",
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:12px;color:gray;'>"
    "DSBDAL Mini Project | TE IT B | AY 2025-26 | "
    "Dataset: Maternal Health Risk | App: Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
