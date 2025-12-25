from pathlib import Path
import pandas as pd
import json
import joblib
import streamlit as st

@st.cache_resource
def load_artifacts():
    base = Path(__file__).resolve().parent
    model_path = base / "rf_animal_condition.joblib"
    meta_path  = base / "metadata.json"

    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_artifacts()

st.title("Animal Condition Prediction (Random Forest)")

st.write("Enter feature values and click **Predict**.")

with st.form("predict_form"):
    inputs = {}

    # Numeric inputs
    for col in meta["numeric_cols"]:
        minv = meta["numeric_min"][col]
        maxv = meta["numeric_max"][col]
        default = meta["numeric_median"][col]
        # number_input avoids slider issues if range is huge
        inputs[col] = st.number_input(
            label=col,
            value=float(default),
            min_value=float(minv),
            max_value=float(maxv),
        )

    # Categorical inputs
    for col in meta["categorical_cols"]:
        options = meta["categorical_values"][col]
        # If there are many unique values, still works; selectbox will be long.
        inputs[col] = st.selectbox(col, options=options)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a single-row dataframe with the same feature columns order used in training
    row = {c: inputs.get(c) for c in meta["feature_cols"]}
    X_input = pd.DataFrame([row])

    pred = model.predict(X_input)[0]
    st.subheader("Prediction")
    st.write(f"**{pred}**")

    # Show probabilities if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        prob_df = pd.DataFrame({"class": meta["classes"], "probability": probs}).sort_values(
            "probability", ascending=False
        )
        st.subheader("Class probabilities")
        st.dataframe(prob_df, use_container_width=True)

st.divider()

st.subheader("Batch prediction (CSV upload)")
uploaded = st.file_uploader("Upload a CSV with the same feature columns", type=["csv"])
if uploaded is not None:
    batch = pd.read_csv(uploaded)
    # keep only expected columns (and in correct order)
    batch = batch[meta["feature_cols"]]
    batch_pred = model.predict(batch)

    out = batch.copy()
    out["prediction"] = batch_pred
    st.dataframe(out, use_container_width=True)

    st.download_button(
        "Download predictions as CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",

    )

