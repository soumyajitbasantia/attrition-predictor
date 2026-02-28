import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# --- Page Configuration ---
st.set_page_config(page_title="HR Attrition Predictor Pro", layout="wide")

@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv("employee_attrition_dataset.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'employee_attrition_dataset.csv' is in the same folder.")
        st.stop()

    # 1. Clean Data: Drop columns with only 1 unique value (fixes the slider error)
    df = df.drop([col for col in df.columns if df[col].nunique() <= 1], axis=1)

    # 2. Target Encoding
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # 3. Define Features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Models
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42))
    ])

    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    # Training
    rf_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)

    # 5. Metrics Calculation
    def get_metrics(model, X_t, y_t):
        preds = model.predict(X_t)
        probs = model.predict_proba(X_t)[:, 1]
        return {
            "acc": accuracy_score(y_t, preds),
            "f1": f1_score(y_t, preds),
            "auc": roc_auc_score(y_t, probs),
            "preds": preds
        }

    rf_m = get_metrics(rf_pipeline, X_test, y_test)
    lr_m = get_metrics(lr_pipeline, X_test, y_test)

    # Determine Best Model based on ROC-AUC
    if rf_m['auc'] >= lr_m['auc']:
        best_model, best_name, best_preds = rf_pipeline, "Random Forest", rf_m['preds']
    else:
        best_model, best_name, best_preds = lr_pipeline, "Logistic Regression", lr_m['preds']

    metrics_dict = {
        "RF": rf_m, "LR": lr_m, "Best Name": best_name
    }

    return best_model, metrics_dict, X, numeric_features, categorical_features, df, (y_test, best_preds)

# --- Initialize Data ---
model_pipeline, metrics, X_data, num_cols, cat_cols, df_full, test_results = load_and_train()

st.title("📊 HR Attrition Analytics & Prediction")

tab1, tab2 = st.tabs(["Individual Prediction", "Model Performance & Insights"])

# --- TAB 1: INDIVIDUAL PREDICTION ---
with tab1:
    st.sidebar.header("Input Employee Features")
    user_inputs = {}

    for col in X_data.columns:
        if col in num_cols:
            min_v, max_v = float(X_data[col].min()), float(X_data[col].max())
            mean_v = float(X_data[col].mean())
            # Safer slider logic
            if min_v < max_v:
                user_inputs[col] = st.sidebar.slider(col, min_v, max_v, mean_v)
            else:
                user_inputs[col] = st.sidebar.number_input(col, value=min_v, disabled=True)
        else:
            user_inputs[col] = st.sidebar.selectbox(col, X_data[col].unique())

    input_df = pd.DataFrame([user_inputs])
    
    st.subheader("Current Employee Profile")
    st.dataframe(input_df)

    if st.button("Analyze Retention Risk", use_container_width=True):
        prob = model_pipeline.predict_proba(input_df)[0][1]
        
        st.divider()
        col_res, col_gauge = st.columns(2)
        
        with col_res:
            if prob > 0.5:
                st.error("### Prediction: High Attrition Risk")
                st.write("This employee is statistically likely to leave the company.")
            else:
                st.success("### Prediction: Low Attrition Risk")
                st.write("This employee is statistically likely to stay.")
        
        with col_gauge:
            st.metric("Probability of Leaving", f"{prob:.2%}")
            st.progress(prob)

# --- TAB 2: PERFORMANCE & INSIGHTS ---
with tab2:
    st.header("Model Evaluation Metrics")
    
    # Metrics Table
    perf_df = pd.DataFrame({
        "Metric": ["Accuracy (Overall Correctness)", "F1-Score (Model Balance)", "ROC-AUC (Separation Power)"],
        "Random Forest": [f"{metrics['RF']['acc']:.2%}", f"{metrics['RF']['f1']:.2f}", f"{metrics['RF']['auc']:.2f}"],
        "Logistic Regression": [f"{metrics['LR']['acc']:.2%}", f"{metrics['LR']['f1']:.2f}", f"{metrics['LR']['auc']:.2f}"]
    })
    st.table(perf_df)
    
    st.info(f"The system is currently using the **{metrics['Best Name']}** model for predictions.")

    col_cm, col_feat = st.columns(2)
    
    with col_cm:
        st.subheader("Confusion Matrix")
        y_test, y_pred = test_results
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Stay", "Leave"], cmap='Blues', ax=ax)
        st.pyplot(fig)
        st.caption("Shows how many employees the model correctly identified as staying or leaving.")

    with col_feat:
        if metrics["Best Name"] == "Random Forest":
            st.subheader("Top Predictors of Attrition")
            raw_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
            clean_names = [n.split('__')[-1] for n in raw_names]
            importances = model_pipeline.named_steps['classifier'].feature_importances_
            feat_imp = pd.Series(importances, index=clean_names).sort_values(ascending=False).head(10)
            st.bar_chart(feat_imp)
            st.caption("Factors that contribute most to the model's decision.")

    st.divider()
    st.subheader("Raw Data Preview")
    st.dataframe(df_full.head(10))
