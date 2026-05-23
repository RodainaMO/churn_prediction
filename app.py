import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Page config 
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero p  { font-size: 1rem;  opacity: 0.8;     margin: 0.4rem 0 0; }

    /* ── KPI cards ── */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #4C9BE8;
        margin-bottom: 1rem;
    }
    .kpi-card.red   { border-left-color: #E8604C; }
    .kpi-card.green { border-left-color: #6BCB77; }
    .kpi-card h3 { margin: 0; font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-card h2 { margin: 0.2rem 0 0; font-size: 1.8rem; font-weight: 700; }

    /* ── Risk badge ── */
    .risk-high  { background:#E8604C; color:white; border-radius:8px; padding:1rem 1.5rem; text-align:center; }
    .risk-med   { background:#FFD166; color:#333;  border-radius:8px; padding:1rem 1.5rem; text-align:center; }
    .risk-low   { background:#6BCB77; color:white; border-radius:8px; padding:1rem 1.5rem; text-align:center; }
    .risk-high h2, .risk-med h2, .risk-low h2 { font-size:1.6rem; margin:0; }
    .risk-high p,  .risk-med p,  .risk-low p  { margin:0.2rem 0 0; font-size:0.9rem; opacity:0.9; }

    /* ── Section headers ── */
    .section-title {
        font-size: 1.1rem; font-weight: 700;
        color: #1a1a2e; border-bottom: 2px solid #4C9BE8;
        padding-bottom: 0.3rem; margin-bottom: 1rem;
    }
    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

#  Load / train model 
@st.cache_resource(show_spinner="Training model on IBM Telco dataset…")
def load_or_train_model():
    

    if os.path.exists("churn_model.pkl"):
        with open("churn_model.pkl", "rb") as f:
            return pickle.load(f)

    # ── Train from scratch ────────────────────────────────────────────────
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    url = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
           "/master/data/Telco-Customer-Churn.csv")
    df = pd.read_csv(url)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Feature engineering
    df["AvgMonthlySpend"]   = df["TotalCharges"] / (df["tenure"] + 1)
    df["HasMultiServices"]  = (
        (df["PhoneService"] == "Yes").astype(int) +
        (df["InternetService"] != "No").astype(int) +
        (df["OnlineSecurity"] == "Yes").astype(int) +
        (df["TechSupport"] == "Yes").astype(int)
    )
    df["IsSeniorLongTenure"] = ((df["SeniorCitizen"] == 1) & (df["tenure"] > 24)).astype(int)
    df["TenureBucket"] = pd.cut(df["tenure"], bins=[0,12,24,48,72],
                                labels=["0-1yr","1-2yr","2-4yr","4+yr"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    num_features = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object","category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_features),
        ("cat", categorical_pipeline, cat_features)
    ])

    model = ImbPipeline([
        ("prep",  preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf",   XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                use_label_encoder=False, eval_metric="logloss",
                                random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    best_thresh = float(thresholds[f1_scores[:-1].argmax()])

    artifacts = {
        "model": model,
        "num_features": num_features,
        "cat_features": cat_features,
        "threshold": best_thresh,
        "test_auc": float(test_auc),
    }
    with open("churn_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    return artifacts

artifacts   = load_or_train_model()
model       = artifacts["model"]
num_features = artifacts["num_features"]
cat_features = artifacts["cat_features"]
THRESHOLD   = artifacts["threshold"]
TEST_AUC    = artifacts["test_auc"]

# Hero 
st.markdown("""
<div class="hero">
  <h1>📊 Customer Churn Predictor</h1>
  <p>AI-powered early warning system · IBM Telco Dataset · XGBoost + SMOTE</p>
</div>
""", unsafe_allow_html=True)

# Tabs 
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction", "📈 Model Insights"])

# ═══════════════════════════ TAB 1 : SINGLE ══════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Enter Customer Profile</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
        st.markdown("### ⚙️ Model Settings")
        threshold = st.slider("Decision Threshold", 0.1, 0.9,
                              float(THRESHOLD), 0.01,
                              help="Lower = catch more churners (higher recall). Higher = fewer false alarms.")
        st.info(f"Model Test AUC: **{TEST_AUC:.3f}**")
        st.markdown("---")
        st.markdown("**About**\nBuilt with XGBoost + SMOTE on IBM Telco Churn dataset (7,043 customers).")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        gender        = st.selectbox("Gender",           ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen",  ["No", "Yes"])
        partner       = st.selectbox("Partner",          ["Yes", "No"])
        dependents    = st.selectbox("Dependents",       ["No", "Yes"])
        tenure        = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("**📡 Services**")
        phone_service     = st.selectbox("Phone Service",     ["Yes", "No"])
        multiple_lines    = st.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
        internet_service  = st.selectbox("Internet Service",  ["Fiber optic", "DSL", "No"])
        online_security   = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
        online_backup     = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support      = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
        streaming_tv      = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
        streaming_movies  = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**💳 Billing**")
        contract       = st.selectbox("Contract",        ["Month-to-month", "One year", "Two year"])
        paperless      = st.selectbox("Paperless Billing",["Yes", "No"])
        payment_method = st.selectbox("Payment Method",  [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
        total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                          float(monthly_charges * tenure), 10.0)

    # ── Build input dataframe ─────────────────────────────────────────────
    input_dict = {
        "gender"          : gender,
        "SeniorCitizen"   : 1 if senior_citizen == "Yes" else 0,
        "Partner"         : partner,
        "Dependents"      : dependents,
        "tenure"          : tenure,
        "PhoneService"    : phone_service,
        "MultipleLines"   : multiple_lines,
        "InternetService" : internet_service,
        "OnlineSecurity"  : online_security,
        "OnlineBackup"    : online_backup,
        "DeviceProtection": device_protection,
        "TechSupport"     : tech_support,
        "StreamingTV"     : streaming_tv,
        "StreamingMovies" : streaming_movies,
        "Contract"        : contract,
        "PaperlessBilling": paperless,
        "PaymentMethod"   : payment_method,
        "MonthlyCharges"  : monthly_charges,
        "TotalCharges"    : total_charges,
    }
    input_df = pd.DataFrame([input_dict])

    # Feature engineering (must mirror notebook)
    input_df["AvgMonthlySpend"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)
    input_df["HasMultiServices"] = (
        (input_df["PhoneService"] == "Yes").astype(int) +
        (input_df["InternetService"] != "No").astype(int) +
        (input_df["OnlineSecurity"] == "Yes").astype(int) +
        (input_df["TechSupport"] == "Yes").astype(int)
    )
    input_df["IsSeniorLongTenure"] = (
        (input_df["SeniorCitizen"] == 1) & (input_df["tenure"] > 24)).astype(int)
    input_df["TenureBucket"] = pd.cut(input_df["tenure"], bins=[0,12,24,48,72],
                                       labels=["0-1yr","1-2yr","2-4yr","4+yr"])

    # ── Predict ───────────────────────────────────────────────────────────
    if st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary"):
        prob  = model.predict_proba(input_df)[0][1]
        label = "CHURN" if prob >= threshold else "RETAIN"

        st.markdown("---")
        r1, r2, r3 = st.columns([1,2,1])
        with r2:
            if prob >= 0.65:
                css, emoji, msg = "risk-high", "🔴", "High Risk — Immediate Action Needed"
            elif prob >= 0.35:
                css, emoji, msg = "risk-med",  "🟡", "Medium Risk — Monitor Closely"
            else:
                css, emoji, msg = "risk-low",  "🟢", "Low Risk — Customer Likely Retained"
            st.markdown(
                f'<div class="{css}"><h2>{emoji} {prob:.1%} Churn Probability</h2>'
                f'<p>{msg}</p></div>', unsafe_allow_html=True)

        st.markdown("####")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="kpi-card {"red" if label=="CHURN" else "green"}">'
                    f'<h3>Decision</h3><h2>{label}</h2></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="kpi-card"><h3>Churn Prob</h3><h2>{prob:.1%}</h2></div>',
                    unsafe_allow_html=True)
        m3.markdown(f'<div class="kpi-card"><h3>Retain Prob</h3><h2>{1-prob:.1%}</h2></div>',
                    unsafe_allow_html=True)
        m4.markdown(f'<div class="kpi-card"><h3>Threshold</h3><h2>{threshold:.2f}</h2></div>',
                    unsafe_allow_html=True)

        # ── Gauge chart ───────────────────────────────────────────────────
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            delta={"reference": threshold * 100, "suffix": "% threshold"},
            title={"text": "Churn Risk Score", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar" : {"color": "#E8604C" if prob >= threshold else "#4C9BE8"},
                "steps": [
                    {"range": [0, 35],  "color": "#e8f5e9"},
                    {"range": [35, 65], "color": "#fff9e6"},
                    {"range": [65, 100],"color": "#fdecea"},
                ],
                "threshold": {
                    "line": {"color": "#333", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # ── Recommendations ───────────────────────────────────────────────
        if label == "CHURN":
            st.markdown("###  Recommended Retention Actions")
            recs = []
            if contract == "Month-to-month":
                recs.append(" **Offer 1-year contract discount** — biggest churn driver")
            if tenure < 12:
                recs.append(" **Enroll in loyalty onboarding program** — early tenure is highest risk")
            if monthly_charges > 70:
                recs.append(" **Review pricing / offer bundle discount**")
            if tech_support == "No":
                recs.append(" **Offer complimentary Tech Support trial**")
            if not recs:
                recs.append(" **Proactive outreach with personalised offer**")
            for r in recs:
                st.markdown(f"- {r}")

# ═══════════════════════════ TAB 2 : BATCH ═══════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Upload Customer CSV for Batch Prediction</p>',
                unsafe_allow_html=True)
    st.info("Upload a CSV with the same columns as the IBM Telco dataset. "
            "The app will return churn probability for every row.")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        if "customerID" in batch_df.columns:
            ids = batch_df["customerID"]
            batch_df.drop(columns=["customerID"], inplace=True)
        else:
            ids = pd.Series(range(len(batch_df)), name="ID")

        if "Churn" in batch_df.columns:
            batch_df.drop(columns=["Churn"], inplace=True)

        batch_df["TotalCharges"] = pd.to_numeric(batch_df["TotalCharges"], errors="coerce")
        batch_df["TotalCharges"].fillna(batch_df["TotalCharges"].median(), inplace=True)

        # Feature engineering
        batch_df["AvgMonthlySpend"]   = batch_df["TotalCharges"] / (batch_df["tenure"] + 1)
        batch_df["HasMultiServices"]  = (
            (batch_df["PhoneService"] == "Yes").astype(int) +
            (batch_df["InternetService"] != "No").astype(int) +
            (batch_df["OnlineSecurity"] == "Yes").astype(int) +
            (batch_df["TechSupport"] == "Yes").astype(int)
        )
        batch_df["IsSeniorLongTenure"] = (
            (batch_df["SeniorCitizen"] == 1) & (batch_df["tenure"] > 24)).astype(int)
        batch_df["TenureBucket"] = pd.cut(batch_df["tenure"], bins=[0,12,24,48,72],
                                           labels=["0-1yr","1-2yr","2-4yr","4+yr"])

        probs = model.predict_proba(batch_df)[:, 1]
        results_df = pd.DataFrame({
            "CustomerID"      : ids.values,
            "Churn_Probability": probs,
            "Risk_Level"      : pd.cut(probs, bins=[0, 0.35, 0.65, 1.0],
                                       labels=["Low", "Medium", "High"]),
            "Decision"        : ["CHURN" if p >= THRESHOLD else "RETAIN" for p in probs]
        }).sort_values("Churn_Probability", ascending=False)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Customers", len(results_df))
        k2.metric("Predicted Churners", (results_df["Decision"] == "CHURN").sum())
        k3.metric("High Risk",  (results_df["Risk_Level"] == "High").sum())
        k4.metric("Avg Churn Prob", f"{probs.mean():.1%}")

        # Risk distribution donut
        risk_counts = results_df["Risk_Level"].value_counts()
        fig = go.Figure(go.Pie(
            labels=risk_counts.index, values=risk_counts.values,
            hole=0.55,
            marker_colors=["#6BCB77","#FFD166","#E8604C"],
            textinfo="label+percent"
        ))
        fig.update_layout(title="Risk Distribution", height=320,
                          margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            results_df.style
                .background_gradient(subset=["Churn_Probability"], cmap="RdYlGn_r")
                .format({"Churn_Probability": "{:.1%}"}),
            use_container_width=True, height=400
        )
        csv_out = results_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results CSV", csv_out,
                           "churn_predictions.csv", "text/csv")

    else:
        st.markdown("""
        **Expected columns** (same as IBM Telco CSV):
        `gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges`
        """)

# ═══════════════════════════ TAB 3 : INSIGHTS ════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Model Performance & Feature Insights</p>',
                unsafe_allow_html=True)

    # ── Load dataset for EDA charts ───────────────────────────────────────
    @st.cache_data
    def load_data():
        url = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
               "/master/data/Telco-Customer-Churn.csv")
        df = pd.read_csv(url)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        return df

    eda_df = load_data()

    # Performance KPIs
    p1, p2, p3 = st.columns(3)
    p1.markdown(f'<div class="kpi-card green"><h3>ROC-AUC</h3><h2>{TEST_AUC:.3f}</h2></div>',
                unsafe_allow_html=True)
    p2.markdown(f'<div class="kpi-card"><h3>Algorithm</h3><h2>XGBoost</h2></div>',
                unsafe_allow_html=True)
    p3.markdown(f'<div class="kpi-card"><h3>Best Threshold</h3><h2>{THRESHOLD:.2f}</h2></div>',
                unsafe_allow_html=True)

    # ── Feature importances ───────────────────────────────────────────────
    clf       = model.named_steps["clf"]
    all_feats = num_features + cat_features
    imp_df    = pd.DataFrame({
        "Feature"   : all_feats,
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=True).tail(12)

    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues",
                 title="Top Feature Importances (XGBoost)")
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                      height=420, margin=dict(t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # ── Churn rate by Contract ────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        cr = eda_df.groupby("Contract")["Churn"].mean().reset_index()
        cr["Churn"] = cr["Churn"] * 100
        fig2 = px.bar(cr, x="Contract", y="Churn",
                      color="Churn", color_continuous_scale="Reds",
                      title="Churn Rate by Contract Type",
                      labels={"Churn": "Churn Rate (%)"})
        fig2.update_layout(coloraxis_showscale=False, height=340)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        cr2 = eda_df.groupby("InternetService")["Churn"].mean().reset_index()
        cr2["Churn"] = cr2["Churn"] * 100
        fig3 = px.bar(cr2, x="InternetService", y="Churn",
                      color="Churn", color_continuous_scale="Blues",
                      title="Churn Rate by Internet Service",
                      labels={"Churn": "Churn Rate (%)"})
        fig3.update_layout(coloraxis_showscale=False, height=340)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tenure distribution ────────────────────────────────────────────────
    fig4 = px.histogram(eda_df, x="tenure", color=eda_df["Churn"].map({1:"Churn",0:"No Churn"}),
                        barmode="overlay", nbins=40,
                        color_discrete_map={"Churn":"#E8604C","No Churn":"#4C9BE8"},
                        title="Tenure Distribution by Churn Status",
                        labels={"color": "Status", "tenure": "Tenure (months)"})
    fig4.update_layout(height=340)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    ---
    ###  Key Business Insights
    | Driver | Finding | Action |
    |---|---|---|
    | **Contract** | Month-to-month → 43% churn vs <5% for 2-yr | Incentivise long-term contracts |
    | **Tenure** | First 12 months: highest risk period | Loyalty onboarding programme |
    | **Monthly Charges** | High charges correlate with churn | Review pricing; offer bundles |
    | **Tech Support** | No support → significantly higher churn | Promote add-on during onboarding |
    | **Internet (Fiber)** | Fibre users churn more than DSL | Investigate service quality |
    """)
