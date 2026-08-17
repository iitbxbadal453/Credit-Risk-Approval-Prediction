import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

bundle = pickle.load(open('credit_risk_model.pkl', 'rb'))

model = bundle["model"]
label_encoder = bundle["label_encoder"]
feature_columns = bundle["feature_columns"]
scalers = bundle["scalers"]
numeric_columns_to_scale = bundle["numeric_columns_to_scale"]
numeric_input_columns = bundle["numeric_input_columns"]
categorical_columns = bundle["categorical_columns"]
categorical_values = bundle["categorical_values"]
education_categories = bundle["education_categories"]
education_mapping = bundle["education_mapping"]

st.title("💳 Credit Risk Approval Predictor")
st.write(
    "Enter an applicant's details below to predict the approval flag "
)

with st.form("prediction_form"):
    st.subheader("Numeric details")

    # Two columns just to keep a long list of numeric fields compact
    numeric_inputs = {}
    cols = st.columns(4)
    for i, col_name in enumerate(numeric_input_columns):
        with cols[i % 4]:
            numeric_inputs[col_name] = st.number_input(
                col_name, value=0.0, step=1.0, format="%.4f"
            )

    st.subheader("Categorical details")
    categorical_inputs = {}
    for col_name in categorical_columns:
        options = categorical_values.get(col_name, [])
        categorical_inputs[col_name] = st.selectbox(col_name, options)

    education = st.selectbox("EDUCATION", education_categories)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Start with every model feature at 0, then fill in what the user gave us.
    row = {col: 0 for col in feature_columns}

    for col_name, value in numeric_inputs.items():
        if col_name in row:
            row[col_name] = value

    # EDUCATION is ordinal-encoded, not one-hot encoded.
    row["EDUCATION"] = education_mapping.get(education, 1)

    for col_name in categorical_columns:
        value = categorical_inputs[col_name]
        dummy_col = f"{col_name}_{value}"
        if dummy_col in row:
            row[dummy_col] = 1

    input_df = pd.DataFrame([row], columns=feature_columns)

    # Apply the same per-column scalers fitted during training.
    for col_name in numeric_columns_to_scale:
        if col_name in input_df.columns and col_name in scalers:
            input_df[col_name] = scalers[col_name].transform(input_df[[col_name]])

    prediction_encoded = model.predict(input_df)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    st.success(f"Predicted Approval Flag: **{prediction_label}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        proba_df = pd.DataFrame(
            {"Class": label_encoder.classes_, "Probability": proba}
        ).sort_values("Probability", ascending=False)
        st.subheader("Class probabilities")
        st.dataframe(proba_df, use_container_width=True, hide_index=True)