
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 0;
}
.subtitle {
    color: #6b7280;
    font-size: 16px;
    margin-bottom: 25px;
}
.metric-card {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
}
.risk-high {
    color: #dc2626;
    font-weight: 700;
    font-size: 24px;
}
.risk-low {
    color: #16a34a;
    font-weight: 700;
    font-size: 24px;
}
</style>
""", unsafe_allow_html=True)

DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

DROP_FEATURES = [
    "PhoneService",
    "gender",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
    "InternetService"
]

# These are the same four base models used in the supplied project.
@st.cache_resource
def train_project():
    data = pd.read_csv(DATA_FILE)

    # Match the supplied preprocessing
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(0)

    original_with_id = data.copy()

    model_df = data.drop(columns=["customerID"]).copy()

    # Encode categorical columns exactly for model training.
    encoders = {}
    for col in model_df.columns:
        if model_df[col].dtype == "object":
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            encoders[col] = le

    # Scale the three numerical variables as in the supplied project.
    scalers = {}
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        scaler = MinMaxScaler()
        model_df[col] = scaler.fit_transform(model_df[[col]])
        scalers[col] = scaler

    model_df = model_df.drop(columns=DROP_FEATURES)

    X = model_df.drop(columns=["Churn"])
    y = model_df["Churn"]

    # Same SMOTE strategy used in the supplied project.
    smote = SMOTE(sampling_strategy=1, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res,
        y_res,
        test_size=0.20,
        random_state=2,
        stratify=y_res
    )

    xgb = XGBClassifier(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=1000,
        random_state=42,
        eval_metric="logloss"
    )

    lgbm = LGBMClassifier(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=1000,
        verbosity=-1,
        random_state=42
    )

    rf = RandomForestClassifier(
        max_depth=4,
        random_state=0
    )

    dt = DecisionTreeClassifier(
        random_state=1000,
        max_depth=4,
        min_samples_leaf=1
    )

    stack = StackingClassifier(
        estimators=[
            ("classifier_xgb", xgb),
            ("classifier_lgbm", lgbm),
            ("classifier_rf", rf),
            ("classifier_dt", dt)
        ],
        final_estimator=lgbm
    )

    models = {
        "Decision Tree": dt,
        "Random Forest": rf,
        "LightGBM": lgbm,
        "XGBoost": xgb,
        "Stacking": stack
    }

    results = []

    cv = RepeatedStratifiedKFold(
        n_splits=10,
        n_repeats=3,
        random_state=42
    )

    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        cv_score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        ).mean()

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
            "CV ROC-AUC": cv_score
        })

        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values(
        "F1 Score", ascending=False
    )

    return {
        "raw_data": original_with_id,
        "encoders": encoders,
        "scalers": scalers,
        "features": list(X.columns),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": fitted_models,
        "results": results_df
    }


def encode_customer(customer, artifact):
    """Convert one raw customer row into the exact 13 model features."""
    row = customer.copy()

    # Work on a one-row DataFrame so the fitted scalers are reused.
    temp = pd.DataFrame([row])

    for col, encoder in artifact["encoders"].items():
        if col in temp.columns:
            value = str(temp.loc[0, col])
            if value not in encoder.classes_:
                raise ValueError(
                    f"Unknown value '{value}' for column '{col}'."
                )
            temp[col] = encoder.transform([value])

    for col, scaler in artifact["scalers"].items():
        temp[col] = scaler.transform(temp[[col]])

    temp = temp.drop(columns=DROP_FEATURES, errors="ignore")
    temp = temp.drop(columns=["Churn", "customerID"], errors="ignore")

    return temp[artifact["features"]]


def risk_label(prob):
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    return "Low Risk"


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
try:
    artifact = train_project()
except FileNotFoundError:
    st.error(
        f"Dataset not found. Put '{DATA_FILE}' in the same folder as app.py."
    )
    st.stop()

data = artifact["raw_data"]
models = artifact["models"]
results_df = artifact["results"]

# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
st.sidebar.title("📊 Telecom Churn")
page = st.sidebar.radio(
    "Dashboard",
    [
        "Overview",
        "Customer Prediction",
        "Model Performance",
        "Churn Analysis",
        "Explainability"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Stacking-based Telecom Churn Prediction")
st.sidebar.caption("XGBoost + LightGBM + Random Forest + Decision Tree")

# ------------------------------------------------------------
# Overview
# ------------------------------------------------------------
if page == "Overview":
    st.markdown('<div class="main-title">Telecom Customer Churn Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Interactive dashboard based on the supplied churn prediction project.</div>',
        unsafe_allow_html=True
    )

    total_customers = len(data)
    churned = int((data["Churn"] == "Yes").sum())
    retained = total_customers - churned
    churn_rate = churned / total_customers

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Retained Customers", f"{retained:,}")
    c4.metric("Churn Rate", f"{churn_rate:.1%}")

    st.markdown("### Customer Churn Overview")

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = data["Churn"].value_counts()
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title("Churn Distribution")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}",
                "ROC-AUC": "{:.2%}",
                "CV ROC-AUC": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "The supplied project uses SMOTE to balance the target before model training "
        "and compares Decision Tree, Random Forest, LightGBM, XGBoost and a Stacking model."
    )

# ------------------------------------------------------------
# Customer Prediction
# ------------------------------------------------------------
elif page == "Customer Prediction":
    st.markdown('<div class="main-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Enter customer details and predict the probability of churn.</div>',
        unsafe_allow_html=True
    )

    customer_ids = data["customerID"].astype(str).tolist()

    selected_id = st.selectbox(
        "Select an existing customer",
        customer_ids
    )

    customer = data[data["customerID"].astype(str) == selected_id].iloc[0]

    with st.expander("Customer Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"**Customer ID:** {customer['customerID']}")
            st.write(f"**Gender:** {customer['gender']}")
            st.write(f"**Senior Citizen:** {customer['SeniorCitizen']}")
            st.write(f"**Partner:** {customer['Partner']}")
            st.write(f"**Dependents:** {customer['Dependents']}")
            st.write(f"**Tenure:** {customer['tenure']} months")

        with col2:
            st.write(f"**Contract:** {customer['Contract']}")
            st.write(f"**Paperless Billing:** {customer['PaperlessBilling']}")
            st.write(f"**Payment Method:** {customer['PaymentMethod']}")
            st.write(f"**Monthly Charges:** ${float(customer['MonthlyCharges']):.2f}")
            st.write(f"**Total Charges:** ${float(customer['TotalCharges']):.2f}")

        with col3:
            st.write(f"**Online Security:** {customer['OnlineSecurity']}")
            st.write(f"**Online Backup:** {customer['OnlineBackup']}")
            st.write(f"**Device Protection:** {customer['DeviceProtection']}")
            st.write(f"**Tech Support:** {customer['TechSupport']}")

    if st.button("🔍 Predict Churn", type="primary", use_container_width=True):
        try:
            X_customer = encode_customer(customer, artifact)

            stack = models["Stacking"]
            prediction = int(stack.predict(X_customer)[0])
            probability = float(stack.predict_proba(X_customer)[0, 1])

            risk = risk_label(probability)

            st.markdown("### Prediction Result")

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Churn Probability",
                f"{probability:.1%}"
            )

            r2.metric(
                "Prediction",
                "Likely to Churn" if prediction == 1 else "Likely to Stay"
            )

            r3.metric("Risk Level", risk)

            if risk == "High Risk":
                st.error("⚠️ High churn risk. Consider proactive retention action.")
            elif risk == "Medium Risk":
                st.warning("⚠️ Medium churn risk. Customer should be monitored.")
            else:
                st.success("✅ Low churn risk. Customer is likely to stay.")

            # Show component-model probabilities.
            st.markdown("### Model Predictions")

            component_names = [
                "XGBoost",
                "LightGBM",
                "Random Forest",
                "Decision Tree",
                "Stacking"
            ]

            probs = []
            for name in component_names:
                p = float(models[name].predict_proba(X_customer)[0, 1])
                probs.append({"Model": name, "Churn Probability": p})

            prob_df = pd.DataFrame(probs)
            st.bar_chart(
                prob_df.set_index("Model")["Churn Probability"]
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ------------------------------------------------------------
# Model Performance
# ------------------------------------------------------------
elif page == "Model Performance":
    st.markdown('<div class="main-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Comparison of all models used in the project.</div>',
        unsafe_allow_html=True
    )

    display_df = results_df.copy()

    st.dataframe(
        display_df.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "ROC-AUC": "{:.2%}",
            "CV ROC-AUC": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Metric Comparison")

    metric = st.selectbox(
        "Select metric",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "CV ROC-AUC"]
    )

    chart_df = results_df.set_index("Model")[[metric]]
    st.bar_chart(chart_df)

    st.markdown("### Stacking Model Evaluation")

    stack = models["Stacking"]
    y_test = artifact["y_test"]
    y_pred = stack.predict(artifact["X_test"])
    y_prob = stack.predict_proba(artifact["X_test"])[:, 1]

    left, right = st.columns(2)

    with left:
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Stacking Confusion Matrix")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            ax=ax
        )
        ax.set_title("Stacking ROC Curve")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    PrecisionRecallDisplay.from_predictions(
        y_test,
        y_prob,
        ax=ax
    )
    ax.set_title("Stacking Precision-Recall Curve")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Classification Report")
    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True
    )
    st.dataframe(
        pd.DataFrame(report).T.round(3),
        use_container_width=True
    )

# ------------------------------------------------------------
# Churn Analysis
# ------------------------------------------------------------
elif page == "Churn Analysis":
    st.markdown('<div class="main-title">Churn Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Explore the categorical and numerical patterns identified in the project.</div>',
        unsafe_allow_html=True
    )

    categorical_options = [
        "Contract",
        "PaymentMethod",
        "PaperlessBilling",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "SeniorCitizen",
        "Partner",
        "Dependents"
    ]

    selected_feature = st.selectbox(
        "Select feature",
        categorical_options
    )

    grouped = (
        data.groupby([selected_feature, "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    pivot = grouped.pivot(
        index=selected_feature,
        columns="Churn",
        values="Customers"
    ).fillna(0)

    st.bar_chart(pivot)

    st.markdown("### Churn Rate by Selected Feature")

    churn_rate_table = (
        data.groupby(selected_feature)["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .sort_values(ascending=False)
        .to_frame("Churn Rate")
    )

    st.dataframe(
        churn_rate_table.style.format({"Churn Rate": "{:.2%}"}),
        use_container_width=True
    )

    st.markdown("### Numerical Features")

    numerical_feature = st.selectbox(
        "Select numerical feature",
        ["tenure", "MonthlyCharges", "TotalCharges"]
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(
        data=data,
        x=numerical_feature,
        hue="Churn",
        kde=True,
        ax=ax
    )
    ax.set_title(f"{numerical_feature} Distribution by Churn")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ------------------------------------------------------------
# Explainability
# ------------------------------------------------------------
elif page == "Explainability":
    st.markdown('<div class="main-title">Customer Explainability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">SHAP-based explanation of the XGBoost component of the stacking system.</div>',
        unsafe_allow_html=True
    )

    try:
        import shap
    except ImportError:
        st.error("SHAP is not installed. Add 'shap' to requirements.txt.")
        st.stop()

    selected_id = st.selectbox(
        "Select customer",
        data["customerID"].astype(str).tolist(),
        key="shap_customer"
    )

    customer = data[
        data["customerID"].astype(str) == selected_id
    ].iloc[0]

    X_customer = encode_customer(customer, artifact)

    xgb = models["XGBoost"]
    prediction = int(xgb.predict(X_customer)[0])
    probability = float(xgb.predict_proba(X_customer)[0, 1])

    st.metric("XGBoost Churn Probability", f"{probability:.1%}")

    if prediction == 1:
        st.error("XGBoost prediction: Likely to Churn")
    else:
        st.success("XGBoost prediction: Likely to Stay")

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_customer)

    values = np.asarray(shap_values).reshape(-1)

    explanation = pd.DataFrame({
        "Feature": artifact["features"],
        "SHAP Value": values,
        "Model Input": X_customer.iloc[0].values
    })

    explanation["Impact"] = np.where(
        explanation["SHAP Value"] >= 0,
        "Pushes toward Churn",
        "Pushes toward Stay"
    )

    explanation["Absolute Impact"] = explanation["SHAP Value"].abs()

    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False
    )

    st.markdown("### Top Factors")

    top = explanation.head(10).copy()

    st.dataframe(
        top[["Feature", "SHAP Value", "Impact", "Model Input"]].style.format({
            "SHAP Value": "{:.4f}",
            "Model Input": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    plot_df = top.sort_values("SHAP Value")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        plot_df["Feature"],
        plot_df["SHAP Value"]
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("SHAP Value")
    ax.set_title("Top SHAP Factors")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Positive SHAP values push the XGBoost prediction toward churn; "
        "negative values push it toward staying."
    )
