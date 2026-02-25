import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# -----------------------
# Load Models
# -----------------------
model = joblib.load("heart_stacking_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
columns = joblib.load("heart_columns.pkl")

stroke_model = load_model("stroke_nn_model.h5")
stroke_scaler = joblib.load("stroke_scaler.pkl")
stroke_columns = joblib.load("stroke_columns.pkl")

st.set_page_config(page_title="Medical Risk Prediction", layout="centered")

# Sidebar
page = st.sidebar.radio(
    "Navigation",
    ["Machine Learning Description",
     "Heart Disease Prediction",
     "Neural Network Description",
     "Stroke Prediction"]
)

# =====================================================
# PAGE 1 - HEART ML DESCRIPTION
# =====================================================
if page == "Machine Learning Description":

    st.title("🫀 Heart Disease Prediction - Ensemble Machine Learning")

    st.markdown("---")

    st.header("1️⃣ Project Objective")

    st.markdown("""
    This project aims to predict the presence of heart disease 
    using clinical attributes collected from patients.

    The task is Binary Classification:

    0 → No Disease  
    1 → Disease Present

    The original dataset contained severity levels (0–4).
    It was transformed into binary form to align with real-world
    medical screening objectives.
    """)

    st.header("2️⃣ Dataset Information")

    st.markdown("""
    Download Dataset:
    https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

    Feature Types:

    Numerical:
    - age
    - trestbps (resting blood pressure)
    - chol (cholesterol)
    - thalch (max heart rate)
    - oldpeak (ST depression)

    Categorical:
    - sex
    - cp (chest pain type)

    Dataset contains mixed feature types and moderate size.
    """)

    # ================= DETAILED DATA PREP =================
    st.header("3️⃣ Detailed Data Preparation Pipeline")

    st.markdown("""
    Step 1: Removing Irrelevant Features  
    The 'id' column was removed because:
    - It has no predictive meaning
    - It introduces noise
    - It may cause overfitting

    --------------------------------------------------

    Step 2: Target Transformation  
    Original target values: 0–4 (severity levels)

    Converted into:
    - 0 → No Disease
    - 1–4 → Disease

    This simplifies the problem into binary classification
    suitable for screening tasks.

    --------------------------------------------------

    Step 3: Handling Missing Values  

    Numerical Features → Median Imputation

    Reason:
    - Cholesterol and blood pressure may contain outliers
    - Median is robust to extreme values
    - Preserves data distribution

    Categorical Features → Most Frequent Imputation

    Reason:
    - Maintains existing distribution
    - Avoids introducing artificial categories

    --------------------------------------------------

    Step 4: Encoding Categorical Variables  

    One-Hot Encoding was applied.

    Why not Label Encoding?

    Example:
    cp: asymptomatic = 3
    This would imply 3 > 1, which is incorrect.

    One-Hot Encoding prevents artificial ordinal relationships.

    --------------------------------------------------

    Step 5: Feature Scaling  

    StandardScaler applied:

        Z = (X - Mean) / Standard Deviation

    This ensures:
    - Equal feature contribution
    - Proper SVM performance
    - Stable model convergence

    Important:
    Scaler was fitted only on training data
    to prevent Data Leakage.

    --------------------------------------------------

    Step 6: Train-Test Split  

    - 80% Training
    - 20% Testing
    - Random state fixed

    Purpose:
    - Evaluate generalization performance
    - Prevent overfitting
    """)

    st.header("4️⃣ Ensemble Model Architecture")

    st.markdown("""
    Base Models:
    - Random Forest → captures nonlinear relationships
    - SVM → maximizes margin separation
    - Logistic Regression → interpretable linear classifier

    Stacking Process:
    1. Base models generate predictions
    2. Predictions become new features
    3. Logistic Regression learns optimal combination

    This reduces bias and variance simultaneously.
    """)

    st.header("5️⃣ Evaluation")

    st.markdown("""
    Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score

    Achieved Accuracy:
    ~83–84%

    In medical screening, Recall is prioritized
    to reduce False Negatives.
    """)


# =====================================================
# PAGE 2 - HEART PREDICTION
# =====================================================
if page == "Heart Disease Prediction":

    st.title("🧪 Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 100)
        trestbps = st.number_input("Resting Blood Pressure", 80, 200)
        chol = st.number_input("Cholesterol", 100, 600)
        thalch = st.number_input("Max Heart Rate", 60, 220)
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0)

    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type",
                          ["typical angina", "atypical angina",
                           "non-anginal", "asymptomatic"])

    if st.button("Predict"):

        input_dict = {
            "age": age,
            "trestbps": trestbps,
            "chol": chol,
            "thalch": thalch,
            "oldpeak": oldpeak,
            "sex": sex,
            "cp": cp
        }

        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=columns, fill_value=0)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"⚠ High Risk ({probability*100:.2f}%)")
        else:
            st.success(f"✅ Low Risk ({(1-probability)*100:.2f}%)")


# =====================================================
# PAGE 3 - STROKE NN DESCRIPTION
# =====================================================
if page == "Neural Network Description":

    st.title("🧠 Stroke Risk Prediction - Neural Network")

    st.markdown("---")

    st.header("1️⃣ Project Objective")

    st.markdown("""
    This project predicts stroke risk using Deep Learning.

    Binary Classification:
    0 → No Stroke
    1 → Stroke

    Dataset size: 5110 samples
    Highly imbalanced dataset.
    """)

    st.header("2️⃣ Dataset Information")

    st.markdown("""
    Download Dataset:
    https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

    Feature Types:

    Numerical:
    - age
    - avg_glucose_level
    - bmi

    Binary:
    - hypertension
    - heart_disease

    Categorical:
    - gender
    - work_type
    - residence_type
    - smoking_status
    - ever_married
    """)

    st.header("3️⃣ Detailed Data Preparation Pipeline")

    st.markdown("""
    Step 1: Removing ID Column  
    Removed because it has no predictive value.

    --------------------------------------------------

    Step 2: Handling Missing BMI  

    BMI contained missing values.
    Median imputation was applied because:
    - BMI distribution is skewed
    - Median is robust to outliers
    - Prevents data loss

    --------------------------------------------------

    Step 3: Encoding Categorical Variables  

    One-Hot Encoding applied.

    Neural Networks require numerical input.
    One-hot prevents ordinal distortion.

    --------------------------------------------------

    Step 4: Stratified Train-Test Split  

    Because dataset is imbalanced (~5% stroke),
    stratification ensures:
    - Same class ratio in train and test
    - Reliable evaluation

    --------------------------------------------------

    Step 5: Feature Scaling  

    StandardScaler applied.

    Neural networks rely on gradient descent.
    Scaling ensures:
    - Stable gradients
    - Faster convergence
    - Better optimization

    --------------------------------------------------

    Step 6: Handling Class Imbalance  

    Class weights applied during training.

    This increases penalty for misclassifying stroke cases,
    improving Recall and reducing False Negatives.

    --------------------------------------------------

    Step 7: Data Leakage Prevention  

    - Scaler fitted only on training data
    - Test data transformed afterward
    - No information from test used in training
    """)

    st.header("4️⃣ Neural Network Architecture")

    st.markdown("""
    Input: 16 features

    Hidden Layers:
    - 64 neurons (ReLU) + BatchNorm + Dropout
    - 32 neurons (ReLU) + BatchNorm + Dropout
    - 16 neurons (ReLU)

    Output:
    - 1 neuron (Sigmoid)

    Loss: Binary Crossentropy
    Optimizer: Adam
    """)

    st.header("5️⃣ Evaluation")

    st.markdown("""
    Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score

    Medical priority:
    Recall is emphasized to reduce False Negatives.
    """)


# =====================================================
# PAGE 4 - STROKE PREDICTION
# =====================================================
if page == "Stroke Prediction":

    st.title("🧠 Stroke Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0)
        bmi = st.number_input("BMI", 10.0, 60.0)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        work_type = st.selectbox("Work Type",
                                 ["Private", "Self-employed",
                                  "Govt_job", "children"])
        Residence_type = st.selectbox("Residence Type",
                                      ["Urban", "Rural"])
        smoking_status = st.selectbox("Smoking Status",
                                      ["never smoked",
                                       "formerly smoked",
                                       "smokes"])

    if st.button("Predict Stroke Risk"):

        input_dict = {
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "gender": gender,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": Residence_type,
            "smoking_status": smoking_status
        }

        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=stroke_columns, fill_value=0)

        input_scaled = stroke_scaler.transform(input_df)

        prob = stroke_model.predict(input_scaled)[0][0]
        prediction = 1 if prob > 0.5 else 0

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"⚠ High Risk ({prob*100:.2f}%)")
        else:
            st.success(f"✅ Low Risk ({(1-prob)*100:.2f}%)")