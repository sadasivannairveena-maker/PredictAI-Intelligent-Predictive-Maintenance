import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="PredictAI",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("artifacts/predictai_model.joblib")
    threshold = joblib.load("artifacts/threshold.joblib")
    features = joblib.load("artifacts/features.joblib")
    return model, threshold, features

model, threshold, features = load_model()

st.title("⚙️ PredictAI")
st.subheader("Intelligent Predictive Maintenance")
st.caption(
    "AI-powered industrial equipment failure risk estimation"
)

st.divider()

st.sidebar.header("Machine Configuration")

machine_type = st.sidebar.selectbox(
    "Machine Type", ["L", "M", "H"]
)

air_temp = st.sidebar.slider(
    "Air Temperature (K)", 295.0, 305.0, 300.0
)

process_temp = st.sidebar.slider(
    "Process Temperature (K)", 305.0, 315.0, 310.0
)

rpm = st.sidebar.slider(
    "Rotational Speed (RPM)", 1100, 2900, 1500
)

torque = st.sidebar.slider(
    "Torque (Nm)", 0.0, 80.0, 40.0
)

tool_wear = st.sidebar.slider(
    "Tool Wear (minutes)", 0, 260, 100
)

input_data = pd.DataFrame([{
    "Type": machine_type,
    "Air temperature": air_temp,
    "Process temperature": process_temp,
    "Rotational speed": rpm,
    "Torque": torque,
    "Tool wear": tool_wear
}])

input_data = input_data[features]

probability = model.predict_proba(input_data)[0, 1]
prediction = int(probability >= threshold)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Estimated Failure Risk", f"{probability:.1%}")

with col2:
    st.metric(
        "Predicted Status",
        "High Risk" if prediction else "Normal"
    )

with col3:
    st.metric("Decision Threshold", f"{threshold:.0%}")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Machine Health Assessment")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        title={"text": "Estimated Failure Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#EF4444" if prediction else "#10B981"},
            "steps": [
                {"range": [0, 30], "color": "#D1FAE5"},
                {"range": [30, 60], "color": "#FEF3C7"},
                {"range": [60, 100], "color": "#FEE2E2"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": threshold * 100
            }
        }
    ))

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Equipment Status")

    if prediction:
        st.error("⚠️ Elevated Machine Failure Risk")
        st.write(
            "The model estimates an elevated likelihood "
            "of machine failure for these operating conditions."
        )
    else:
        st.success("✅ Normal Operating Condition")
        st.write(
            "The model estimates a lower likelihood of "
            "failure for these operating conditions."
        )

    st.subheader("Current Sensor Readings")
    st.dataframe(input_data, use_container_width=True)

st.divider()

st.subheader("About PredictAI")

st.write(
    "PredictAI uses a Random Forest machine learning "
    "model trained on the UCI AI4I 2020 synthetic "
    "predictive maintenance dataset."
)

st.info(
    "Educational demonstration only. Model predictions "
    "are not validated for real industrial equipment "
    "or safety-critical maintenance decisions."
)
