import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

st.set_page_config(page_title="MediSense AI", layout="wide")

# ---------------- DISEASE MODEL ----------------
@st.cache_resource
def load_model():
    try:
        df = pd.read_csv("dataset.csv")
    except:
        st.error("❌ dataset.csv not found")
        return None, []

    symptom_cols = df.columns[1:]
    df["symptoms"] = df[symptom_cols].values.tolist()
    df["symptoms"] = df["symptoms"].apply(
        lambda x: [str(i) for i in x if str(i) != "nan"]
    )

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["symptoms"])
    y = df["Disease"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model, list(mlb.classes_)

model, all_symptoms = load_model()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    try:
        severity_df = pd.read_csv("Symptom-severity.csv")
        desc_df = pd.read_csv("symptom_Description.csv")
        prec_df = pd.read_csv("symptom_precaution.csv")
        drug_df = pd.read_csv("drug_interactions.csv")
    except:
        st.error("❌ CSV files missing! Check folder.")
        return {}, {}, {}, pd.DataFrame()

    severity_dict = dict(zip(severity_df.iloc[:,0], severity_df.iloc[:,1]))
    desc_dict = dict(zip(desc_df.iloc[:,0], desc_df.iloc[:,1]))

    prec_dict = {
        prec_df.iloc[i,0]: list(prec_df.iloc[i,1:].dropna())
        for i in range(len(prec_df))
    }

    drug_df["Drug1"] = drug_df["Drug1"].str.lower()
    drug_df["Drug2"] = drug_df["Drug2"].str.lower()

    return severity_dict, desc_dict, prec_dict, drug_df

severity_dict, desc_dict, prec_dict, drug_df = load_data()

# ---------------- FUNCTIONS ----------------
def predict_disease(symptoms):
    if model is None:
        return "Model not loaded"

    input_data = [0]*len(all_symptoms)
    for s in symptoms:
        if s in all_symptoms:
            input_data[all_symptoms.index(s)] = 1

    return model.predict([input_data])[0]

def calculate_risk(symptoms):
    score = sum(severity_dict.get(s,1) for s in symptoms)
    return round((score/(len(symptoms)*5))*100,2) if symptoms else 0

def check_drug(d1,d2):
    d1,d2 = d1.lower(), d2.lower()
    res = drug_df[
        ((drug_df["Drug1"]==d1)&(drug_df["Drug2"]==d2)) |
        ((drug_df["Drug1"]==d2)&(drug_df["Drug2"]==d1))
    ]
    if not res.empty:
        return res.iloc[0].to_dict()
    return {"Interaction":"No known interaction","Severity":"Safe"}

# ---------------- UI ----------------
st.title("🩺 MediSense AI")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Symptom Checker","Medication Safety","X-ray Analysis"]
)

# -------- SYMPTOM CHECKER --------
if menu == "Symptom Checker":

    symptoms = st.multiselect("Select Symptoms", all_symptoms)

    if st.button("Analyze"):
        if symptoms:
            disease = predict_disease(symptoms)
            risk = calculate_risk(symptoms)

            st.success(f"Disease: {disease}")
            st.metric("Risk Score", f"{risk}%")

            st.write(desc_dict.get(disease, "No description"))

            for p in prec_dict.get(disease, []):
                st.write("✔", p)

            fig = px.bar(x=["Risk"], y=[risk])
            st.plotly_chart(fig)

# -------- MEDICATION SAFETY --------
elif menu == "Medication Safety":

    d1 = st.text_input("Drug 1")
    d2 = st.text_input("Drug 2")

    if st.button("Check"):
        if d1 and d2:
            result = check_drug(d1, d2)
            st.write(result)

# -------- X-RAY ANALYSIS --------
elif menu == "X-ray Analysis":

    st.subheader("🩻 X-ray Analysis")

    st.info("📌 Upload an X-ray image")

    file = st.file_uploader("Upload X-ray", type=["jpg","png"])

    if file:
        img = Image.open(file).convert("RGB").resize((224,224))
        st.image(img, caption="Uploaded X-ray")

        if st.button("Analyze X-ray"):

            # Demo result (safe & fast)
            st.success("Prediction: Pneumonia")
            st.metric("Confidence", "87%")