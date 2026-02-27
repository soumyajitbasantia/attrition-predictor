import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("employee_attrition_dataset.csv")
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42))
    ])

    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    rf_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)

    rf_acc = accuracy_score(y_test, rf_pipeline.predict(X_test))
    rf_auc = roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:,1])

    lr_acc = accuracy_score(y_test, lr_pipeline.predict(X_test))
    lr_auc = roc_auc_score(y_test, lr_pipeline.predict_proba(X_test)[:,1])

    if rf_auc >= lr_auc:
        best_model = rf_pipeline
        best_name = "Random Forest"
    else:
        best_model = lr_pipeline
        best_name = "Logistic Regression"

    metrics = {
        "RF Accuracy": rf_acc,
        "RF ROC-AUC": rf_auc,
        "LR Accuracy": lr_acc,
        "LR ROC-AUC": lr_auc,
        "Best Model": best_name
    }

    return best_model, metrics, X, numeric_features, categorical_features, df

model_pipeline, metrics, X_data, num_cols, cat_cols, df_full = load_and_train()

st.title("📊 Employee Attrition Prediction Dashboard")

tab1, tab2 = st.tabs(["Prediction", "Dataset Insights"])

with tab1:
    st.sidebar.header("Input Employee Data")
    user_data = {}

    for col in X_data.columns:
        if col in num_cols:
            min_v = float(X_data[col].min())
            max_v = float(X_data[col].max())
            mean_v = float(X_data[col].mean())
            if min_v < max_v:
                user_data[col] = st.sidebar.slider(col, min_v, max_v, mean_v)
            else:
                user_data[col] = st.sidebar.number_input(col, value=min_v)
        else:
            options = X_data[col].unique().tolist()
            user_data[col] = st.sidebar.selectbox(col, options)

    input_df = pd.DataFrame([user_data])
    st.subheader("Selected Employee Summary")
    st.write(input_df)

    st.sidebar.markdown("### Model Performance")
    st.sidebar.write(f"Best Model: {metrics['Best Model']}")
    st.sidebar.write(f"RF Accuracy: {metrics['RF Accuracy']:.2%}")
    st.sidebar.write(f"RF ROC-AUC: {metrics['RF ROC-AUC']:.2f}")
    st.sidebar.write(f"LR Accuracy: {metrics['LR Accuracy']:.2%}")
    st.sidebar.write(f"LR ROC-AUC: {metrics['LR ROC-AUC']:.2f}")

    if st.button("Run Analysis"):
        prediction = model_pipeline.predict(input_df)[0]
        probability = model_pipeline.predict_proba(input_df)[0][1]

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("🚨 High Risk of Attrition (Likely to Leave)")
            else:
                st.success("✅ Low Risk (Likely to Stay)")

        with col2:
            st.metric("Attrition Probability", f"{probability:.2%}")
            st.progress(probability)

        if probability > 0.7:
            st.warning("Very High Risk Category")
        elif probability > 0.4:
            st.info("Moderate Risk Category")
        else:
            st.success("Low Risk Category")

        if metrics["Best Model"] == "Random Forest":
            feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
            importances = pd.Series(
                model_pipeline.named_steps['classifier'].feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)
            st.subheader("Top Contributing Factors")
            st.bar_chart(importances.head(10))

with tab2:
    st.subheader("Dataset Overview")
    st.write(df_full.head())

    attrition_rate = df_full['Attrition'].mean()
    st.metric("Overall Attrition Rate", f"{attrition_rate:.2%}")

    st.subheader("Attrition Distribution")
    st.bar_chart(df_full['Attrition'].value_counts())

    if "Department" in df_full.columns:
        st.subheader("Attrition by Department")
        st.bar_chart(pd.crosstab(df_full['Department'], df_full['Attrition']))

    if "OverTime" in df_full.columns:
        st.subheader("Overtime vs Attrition")
        st.bar_chart(pd.crosstab(df_full['OverTime'], df_full['Attrition']))

    with st.expander("Model Information"):
        st.write("""
        This application trains Logistic Regression and Random Forest models.
        Class imbalance handling is applied.
        Best model is selected using ROC-AUC score.
        Preprocessing includes scaling and one-hot encoding inside a pipeline.
        """)
