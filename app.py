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


# ---------- EXPLAINABLE AI ----------
st.divider()
st.header("🔍 Explainable AI")
st.write(
    "Understand which sensor measurements influence "
    "the current machine failure-risk prediction."
)

if st.button("Explain This Prediction"):

    import shap
    import numpy as np
    import matplotlib.pyplot as plt

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    processed = preprocessor.transform(input_data)

    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    names = preprocessor.get_feature_names_out()

    @st.cache_resource
    def get_shap_explainer(_classifier):
        return shap.TreeExplainer(_classifier)

    explainer = get_shap_explainer(classifier)

    values = explainer.shap_values(processed)

    if isinstance(values, list):
        failure_values = values[1][0]
    else:
        failure_values = values[0, :, 1]

    baseline = float(
        np.asarray(explainer.expected_value)[1]
    )

    explanation = shap.Explanation(
        values=failure_values,
        base_values=baseline,
        data=processed[0],
        feature_names=names
    )

    reconstructed = baseline + np.sum(failure_values)

    if not np.isclose(
        reconstructed, probability, atol=1e-4
    ):
        st.error("SHAP verification failed.")
    else:
        st.success("Explanation verified.")

        fig = plt.figure(figsize=(11, 6))

        shap.plots.waterfall(
            explanation,
            max_display=8,
            show=False
        )

        st.pyplot(plt.gcf())
        plt.close("all")

        st.caption(
            "Red increases the predicted failure score; "
            "blue decreases it. SHAP explains model "
            "behavior, not physical causation."
        )

        contributions = pd.DataFrame({
            "Feature": names,
            "Contribution": failure_values
        })

        contributions["Absolute Impact"] = (
            contributions["Contribution"].abs()
        )

        contributions = contributions.sort_values(
            "Absolute Impact", ascending=False
        )

        st.subheader("Feature Contributions")
        st.dataframe(
            contributions,
            use_container_width=True,
            hide_index=True
        )


# ---------- MODEL PERFORMANCE DASHBOARD ----------

st.divider()
st.header("📊 Model Performance & Evaluation")

st.caption(
    "Exploratory evaluation on the UCI AI4I 2020 "
    "synthetic predictive maintenance dataset."
)

import json
import plotly.express as px
import plotly.graph_objects as go

with open("artifacts/metrics.json", "r") as f:
    metrics = json.load(f)

st.subheader("Evaluation Metrics")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Recall",
        f"{metrics['recall']:.2%}"
    )

with c2:
    st.metric(
        "Precision",
        f"{metrics['precision']:.2%}"
    )

with c3:
    st.metric(
        "F1 Score",
        f"{metrics['f1_score']:.4f}"
    )

c4, c5, c6 = st.columns(3)

with c4:
    st.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}"
    )

with c5:
    st.metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.4f}"
    )

with c6:
    st.metric(
        "PR-AUC",
        f"{metrics['pr_auc']:.4f}"
    )

st.divider()

st.subheader("Confusion Matrix")

cm = metrics["confusion_matrix"]

fig_cm = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["Predicted Normal", "Predicted Failure"],
        y=["Actual Normal", "Actual Failure"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False
    )
)

fig_cm.update_layout(
    title="Machine Failure Classification Results",
    height=420,
    xaxis_title="Predicted Class",
    yaxis_title="Actual Class",
    yaxis={"autorange": "reversed"}
)

st.plotly_chart(fig_cm, width="stretch")

tn, fp = cm[0]
fn, tp = cm[1]

st.write(f"✅ Correctly identified normal records: {tn}")
st.write(f"✅ Detected actual failures: {tp}")
st.write(f"⚠️ Missed failures: {fn}")
st.write(f"⚠️ False failure alarms: {fp}")

st.divider()

st.subheader("Threshold Optimization Comparison")

comparison = pd.DataFrame({
    "Model Setting": [
        "Default threshold (0.50)",
        "Optimized threshold (0.30)"
    ],
    "Detected Failures": [36, 50],
    "Missed Failures": [32, 18],
    "False Alarms": [10, 37]
})

fig_compare = go.Figure()

for column in [
    "Detected Failures",
    "Missed Failures",
    "False Alarms"
]:
    fig_compare.add_trace(
        go.Bar(
            name=column,
            x=comparison["Model Setting"],
            y=comparison[column]
        )
    )

fig_compare.update_layout(
    title="Impact of Classification Threshold",
    barmode="group",
    yaxis_title="Number of Records",
    height=420
)

st.plotly_chart(fig_compare, width="stretch")

st.info(
    "Lowering the threshold increased failure recall "
    "from 52.94% to 73.53%, but false alarms increased "
    "from 10 to 37. This illustrates the trade-off "
    "between missed failures and false alerts."
)

st.divider()

st.subheader("Model Limitations")

st.write(
    """
    - Training data is synthetic, not collected from
      an operating industrial facility.
    - Model scores are not calibrated real-world
      failure probabilities.
    - The classification model does not estimate
      time until equipment failure.
    - Test results were inspected during development
      and require stronger independent validation.
    - The application is a research and educational
      prototype, not a production monitoring system.
    """
)
