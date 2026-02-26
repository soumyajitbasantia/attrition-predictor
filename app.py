import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 1. SETUP & MODEL TRAINING ---
@st.cache_resource # Keeps the model in memory so it doesn't reload every time
def train_model():
    # Load dataset
    df = pd.read_csv("employee_attrition_dataset.csv")
    
    # Map target variable
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Separate Features and Target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Create Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Create Full Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ])
    
    # Train the pipeline
    pipeline.fit(X, y)
    
    return pipeline, X, numeric_features, categorical_features

# Load the trained model and data info
model_pipeline, X_data, num_cols, cat_cols = train_model()

# --- 2. UI LAYOUT ---
st.set_page_config(page_title="HR Attrition Predictor", layout="wide")
st.title("📊 Employee Attrition Prediction Dashboard")
st.markdown("Adjust employee details in the sidebar to see the probability of them leaving.")

st.sidebar.header("Input Employee Data")
user_data = {}

# Automatically create sliders for numbers and dropdowns for categories
for col in X_data.columns:
    if col in num_cols:
        min_v = float(X_data[col].min())
        max_v = float(X_data[col].max())
        mean_v = float(X_data[col].mean())
        
        # FIX: Check if min is different from max to avoid Slider Error
        if min_v < max_v:
            user_data[col] = st.sidebar.slider(f"{col}", min_v, max_v, mean_v)
        else:
            user_data[col] = st.sidebar.number_input(f"{col}", value=min_v)
    else:
        options = X_data[col].unique().tolist()
        user_data[col] = st.sidebar.selectbox(f"{col}", options)

# --- 3. PREDICTION LOGIC ---
st.subheader("Selected Employee Summary")
input_df = pd.DataFrame([user_data])
st.write(input_df)

if st.button("Run Analysis"):
    # The pipeline handles scaling and encoding automatically!
    prediction = model_pipeline.predict(input_df)[0]
    probability = model_pipeline.predict_proba(input_df)[0][1]

    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("🚨 Prediction: High Risk of Attrition (Likely to LEAVE)")
        else:
            st.success("✅ Prediction: Low Risk (Likely to STAY)")
            
    with col2:
        st.metric("Attrition Probability", f"{probability:.2%}")
        st.progress(probability)

    # Feature Importance Visualization
    st.subheader("Top Contributing Factors")
    # Get feature names from the encoder
    feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = pd.Series(model_pipeline.named_steps['classifier'].feature_importances_, index=feature_names).sort_values(ascending=False)
    st.bar_chart(importances.head(10))