import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

st.set_page_config(
    page_title="PredictAI V2",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_resource
def load_resources():
    model = joblib.load(
        "artifacts/gradient_boosting_model.joblib"
    )
    threshold = joblib.load(
        "artifacts/gradient_boosting_threshold.joblib"
    )
    features = joblib.load(
        "artifacts/features.joblib"
    )
    background = joblib.load(
        "artifacts/shap_background.joblib"
    )

    with open(
        "artifacts/gradient_boosting_metrics.json"
    ) as f:
        metrics = json.load(f)

    return model, threshold, features, background, metrics

model, threshold, features, background, metrics = load_resources()

st.title("⚙️ PredictAI V2")
st.subheader("Intelligent Predictive Maintenance")
st.caption("Gradient Boosting · Explainable AI · Model Evaluation")

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
}])[features]

probability = float(model.predict_proba(input_data)[0, 1])
is_high_risk = probability >= threshold

# ---------- PREDICTION ----------
c1, c2, c3 = st.columns(3)

c1.metric("Model Failure Score", f"{probability:.1%}")
c2.metric(
    "Predicted Status",
    "High Risk" if is_high_risk else "Normal"
)
c3.metric("Decision Threshold", f"{threshold:.0%}")

left, right = st.columns(2)

with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {
                "color": "#EF4444" if is_high_risk else "#10B981"
            },
            "threshold": {
                "line": {"color": "black", "width": 4},
                "value": threshold * 100
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Equipment Assessment")
    if is_high_risk:
        st.error("⚠️ Elevated failure-risk score")
    else:
        st.success("✅ Lower failure-risk score")

    st.dataframe(input_data, width="stretch")

# ---------- EXPLAINABLE AI ----------
st.divider()
st.header("🔍 Explainable AI")

if st.button("Explain This Prediction"):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    processed = preprocessor.transform(input_data)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    @st.cache_resource
    def get_explainer():
        return shap.TreeExplainer(
            classifier,
            data=background,
            model_output="probability",
            feature_perturbation="interventional"
        )

    explainer = get_explainer()
    values = np.asarray(explainer.shap_values(processed))
    baseline = float(np.asarray(explainer.expected_value))

    reconstruction = baseline + values[0].sum()
    difference = abs(reconstruction - probability)

    if difference > 0.001:
        st.error(
            "SHAP reconstruction check failed. "
            "Explanation not displayed."
        )
    else:
        st.success("SHAP explanation verified.")

        explanation = shap.Explanation(
            values=values[0],
            base_values=baseline,
            data=np.asarray(processed)[0],
            feature_names=preprocessor.get_feature_names_out()
        )

        shap.plots.waterfall(
            explanation,
            max_display=8,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.close("all")

        st.caption(
            "SHAP contributions explain this model's score, "
            "not the physical causes of machine failure."
        )

# ---------- MODEL EVALUATION ----------
st.divider()
st.header("📊 Model Performance & Evaluation")

c1, c2, c3 = st.columns(3)
c1.metric("Recall", f"{metrics['recall']:.2%}")
c2.metric("Precision", f"{metrics['precision']:.2%}")
c3.metric("F1 Score", f"{metrics['f1_score']:.4f}")

c4, c5, c6 = st.columns(3)
c4.metric("Accuracy", f"{metrics['accuracy']:.2%}")
c5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
c6.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")

st.subheader("Confusion Matrix")

cm = np.asarray(metrics["confusion_matrix"])

heatmap = go.Figure(go.Heatmap(
    z=cm,
    x=["Predicted Normal", "Predicted Failure"],
    y=["Actual Normal", "Actual Failure"],
    text=cm,
    texttemplate="%{text}",
    colorscale="Blues",
    showscale=False
))

heatmap.update_layout(
    height=400,
    yaxis={"autorange": "reversed"},
    xaxis_title="Predicted",
    yaxis_title="Actual"
)
st.plotly_chart(heatmap, width="stretch")

st.info(
    "Results are exploratory, from the synthetic UCI AI4I "
    "benchmark. The test set was inspected during development."
)

st.warning(
    "Educational prototype only. Scores are not calibrated "
    "real-world failure probabilities or validated for "
    "safety-critical maintenance decisions."
)
