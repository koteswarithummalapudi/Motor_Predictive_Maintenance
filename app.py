
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -------------------------------
# 1. Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Motor Health Monitoring",
    page_icon="⚙️",
    layout="wide"
)

# -------------------------------
# 2. Load ML Model
# -------------------------------

@st.cache_resource
def load_model():
    return joblib.load("motor_model.joblib")

model = load_model()

# -------------------------------
# 3. Initialize History
# -------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -------------------------------
# 4. Dashboard Header
# -------------------------------

st.title("⚙️ Motor Health Monitoring System")

st.subheader(
    "Real-Time Motor Health Assessment and Fault Prediction"
)

st.write("IoT and Machine Learning Based Predictive Maintenance")

st.info(
    "Demo Mode: Sensor values are entered manually. "
    "Hardware integration is not completed yet."
)

# -------------------------------
# 5. Sensor Input Section
# -------------------------------

st.header("📡 Motor Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    vibration = st.number_input(
        "Vibration Value",
        min_value=0.0,
        value=0.213,
        step=0.01
    )

with col2:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        value=29.69,
        step=0.5
    )

# -------------------------------
# 6. Prediction
# -------------------------------

if st.button("🔍 Predict Motor Health", type="primary"):

    input_data = pd.DataFrame({
        "Vibration": [vibration],
        "Temperature": [temperature]
    })

    prediction = model.predict(input_data)[0]

    record = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Vibration": vibration,
        "Temperature": temperature,
        "Prediction": prediction
    }

    st.session_state.prediction_history.append(record)

    st.session_state.latest_prediction = prediction

# -------------------------------
# 7. Latest Prediction Result
# -------------------------------

if "latest_prediction" in st.session_state:

    prediction = st.session_state.latest_prediction

    st.divider()
    st.header("🚦 Latest Motor Health Status")

    if prediction == "Normal":
        st.success("Motor Status: NORMAL")

    elif prediction in ["Watch", "Warning"]:
        st.warning(
            f"Attention! Motor Status: {prediction}"
        )

    else:
        st.error(
            f"Alert! Motor Status: {prediction}"
        )

# -------------------------------
# 8. Dashboard Analytics
# -------------------------------

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.divider()
    st.header("📊 Motor Health Summary")

    total = len(history_df)

    counts = history_df["Prediction"].value_counts()

    normal = int(counts.get("Normal", 0))
    watch = int(counts.get("Watch", 0))
    warning = int(counts.get("Warning", 0))
    critical = int(counts.get("Critical", 0))
    failure = int(counts.get("Failure", 0))

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Predictions", total)
    c2.metric("Normal", normal)
    c3.metric("Watch / Warning", watch + warning)
    c4.metric("Critical / Failure", critical + failure)

    # -------------------------------
    # 9. Prediction Distribution
    # -------------------------------

    st.subheader("Motor Health Distribution")

    distribution = counts.reindex(
        ["Normal", "Watch", "Warning", "Critical", "Failure"],
        fill_value=0
    )

    st.bar_chart(distribution)

    # -------------------------------
    # 10. Alert System
    # -------------------------------

    st.divider()
    st.header("🚨 Motor Alert System")

    abnormal_count = watch + warning + critical + failure

    if abnormal_count == 0:
        st.success("All recorded predictions are Normal.")

    else:
        st.warning(
            f"{abnormal_count} recorded predictions "
            "were classified as abnormal. "
            "Check the motor condition."
        )

    # -------------------------------
    # 11. Prediction History
    # -------------------------------

    st.divider()
    st.header("📋 Prediction History")

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # CSV Download
    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="motor_prediction_history.csv",
        mime="text/csv"
    )

    # -------------------------------
    # 12. Vibration Graph
    # -------------------------------

    st.divider()
    st.header("📈 Sensor Reading Analysis")

    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        st.subheader("Vibration Trend")

        st.line_chart(
            history_df["Vibration"]
        )

    with graph_col2:
        st.subheader("Temperature Trend")

        st.line_chart(
            history_df["Temperature"]
        )

    # -------------------------------
    # 13. Clear History
    # -------------------------------

    st.divider()

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []

        if "latest_prediction" in st.session_state:
            del st.session_state.latest_prediction

        st.rerun()

# -------------------------------
# 14. Model Information
# -------------------------------

with st.expander("ℹ️ Model Information"):

    st.write("Model File: motor_model.joblib")

    st.write("Input Features:")
    st.write("- Vibration")
    st.write("- Temperature")

    st.write("Prediction Classes:")
    st.write(
        "Normal, Watch, Warning, Critical, Failure"
    )

    st.write("Previously measured test accuracy: 96.81%")

    st.caption(
        "Accuracy is based on the available dataset test split, "
        "not real-world hardware validation."
    )

# -------------------------------
# 15. Footer
# -------------------------------

st.divider()

st.caption(
    "Motor Predictive Maintenance System | "
    "IoT and Machine Learning | Demo Mode"
)
