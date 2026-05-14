import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Credit Card Fraud Detection")
st.markdown("**Handling Imbalanced Data with SMOTE** — Interactive Analysis Dashboard")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload creditcard.csv (optional)",
    type=["csv"],
    help="Leave blank to use the built-in synthetic dataset.",
)

n_samples = st.sidebar.slider("Synthetic samples", 5_000, 50_000, 10_000, 1_000,
                               disabled=uploaded_file is not None)
fraud_pct = st.sidebar.slider("Synthetic fraud %", 0.5, 5.0, 1.0, 0.5,
                               disabled=uploaded_file is not None)
run_btn = st.sidebar.button("🚀 Train models", type="primary")

# ── Helper: load / generate data ───────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data(file, n_samples, fraud_pct):
    if file is not None:
        df = pd.read_csv(file)
        return df

    np.random.seed(42)
    n_fraud = int(n_samples * fraud_pct / 100)
    n_legit = n_samples - n_fraud

    legit_X  = np.random.randn(n_legit, 28)
    fraud_X  = np.random.randn(n_fraud, 28) * 1.5 + 0.5
    legit_amt  = np.random.uniform(1, 1000, n_legit)
    fraud_amt  = np.random.uniform(100, 5000, n_fraud)

    X = np.vstack([legit_X, fraud_X])
    amounts = np.concatenate([legit_amt, fraud_amt])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = amounts
    df["Class"]  = y
    return df


# ── Helper: train pipeline ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Training models…")
def train_pipeline(df_json):
    import io
    df = pd.read_json(io.StringIO(df_json))

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Baseline (no SMOTE)
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    base_rf.fit(X_train_sc, y_train)
    y_pred_base  = base_rf.predict(X_test_sc)
    y_proba_base = base_rf.predict_proba(X_test_sc)[:, 1]

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train_sc, y_train)
    smote_counts = {"before": y_train.value_counts().to_dict(),
                    "after":  pd.Series(y_bal).value_counts().to_dict()}

    # Random Forest + SMOTE
    rf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                min_samples_split=10, random_state=42, n_jobs=-1)
    rf.fit(X_bal, y_bal)
    y_pred_rf   = rf.predict(X_test_sc)
    y_proba_rf  = rf.predict_proba(X_test_sc)[:, 1]

    # Logistic Regression + SMOTE
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_bal, y_bal)
    y_pred_lr   = lr.predict(X_test_sc)
    y_proba_lr  = lr.predict_proba(X_test_sc)[:, 1]

    # Decision Tree + SMOTE
    dt = DecisionTreeClassifier(max_depth=15, min_samples_split=10, random_state=42)
    dt.fit(X_bal, y_bal)
    y_pred_dt   = dt.predict(X_test_sc)
    y_proba_dt  = dt.predict_proba(X_test_sc)[:, 1]

    models_map = {
        "Baseline RF (No SMOTE)":      (y_pred_base, y_proba_base, base_rf),
        "Random Forest + SMOTE":       (y_pred_rf,   y_proba_rf,   rf),
        "Logistic Regression + SMOTE": (y_pred_lr,   y_proba_lr,   lr),
        "Decision Tree + SMOTE":       (y_pred_dt,   y_proba_dt,   dt),
    }

    rows = []
    for name, (y_p, y_pb, _) in models_map.items():
        rows.append({
            "Model":     name,
            "Accuracy":  accuracy_score(y_test,  y_p),
            "Precision": precision_score(y_test, y_p, zero_division=0),
            "Recall":    recall_score(y_test,    y_p, zero_division=0),
            "F1-Score":  f1_score(y_test,        y_p, zero_division=0),
            "ROC-AUC":   roc_auc_score(y_test,   y_pb),
        })

    comparison = pd.DataFrame(rows).sort_values("Recall", ascending=False)

    feature_imp = pd.DataFrame({
        "Feature":    X.columns.tolist(),
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False).head(10)

    return {
        "comparison":   comparison,
        "y_test":       y_test.tolist(),
        "models_preds": {
            k: {"y_pred": v[0].tolist(), "y_proba": v[1].tolist()}
            for k, v in models_map.items()
        },
        "smote_counts":  smote_counts,
        "feature_imp":   feature_imp.to_dict(),
        "X_columns":     X.columns.tolist(),
        "rf_model":      rf,
        "scaler":        scaler,
    }


# ── Load data (always) ─────────────────────────────────────────────────────────
df = load_data(uploaded_file, n_samples, fraud_pct)

# ── EDA section ────────────────────────────────────────────────────────────────
st.header("1. Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total transactions", f"{len(df):,}")
col2.metric("Legitimate", f"{int((df['Class']==0).sum()):,}")
col3.metric("Fraudulent",  f"{int((df['Class']==1).sum()):,}")
col4.metric("Fraud rate",  f"{df['Class'].mean()*100:.2f}%")

with st.expander("Show raw data sample"):
    st.dataframe(df.head(20), use_container_width=True)

st.subheader("Class distribution & Amount statistics")
fig_eda, axes = plt.subplots(1, 3, figsize=(16, 4))

class_counts = df["Class"].value_counts()
axes[0].bar(["Legitimate", "Fraud"], class_counts.values,
            color=["#3498db", "#e74c3c"], edgecolor="black", alpha=0.8)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for rect in axes[0].patches:
    h = rect.get_height()
    axes[0].text(rect.get_x() + rect.get_width() / 2, h,
                 f"{int(h):,}\n({h/len(df)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

df.boxplot(column="Amount", by="Class", ax=axes[1])
axes[1].set_title("Transaction Amount by Class")
axes[1].set_xlabel("Class (0=Legit, 1=Fraud)")
axes[1].set_ylabel("Amount ($)")
plt.suptitle("")

axes[2].hist(df[df["Class"]==0]["Amount"], bins=50, alpha=0.6, label="Legit",  color="#3498db")
axes[2].hist(df[df["Class"]==1]["Amount"], bins=50, alpha=0.6, label="Fraud",  color="#e74c3c")
axes[2].set_title("Amount Distribution")
axes[2].set_xlabel("Amount ($)")
axes[2].set_ylabel("Frequency")
axes[2].legend()

fig_eda.tight_layout()
st.pyplot(fig_eda)
plt.close(fig_eda)

# ── Training ───────────────────────────────────────────────────────────────────
if run_btn or "results" not in st.session_state:
    with st.spinner("Training…"):
        st.session_state["results"] = train_pipeline(df.to_json())

results = st.session_state.get("results")

if results is None:
    st.info("Click **Train models** in the sidebar to start.")
    st.stop()

comparison   = results["comparison"]
y_test       = np.array(results["y_test"])
models_preds = results["models_preds"]
smote_counts = results["smote_counts"]
feature_imp  = pd.DataFrame(results["feature_imp"])

# ── SMOTE impact ───────────────────────────────────────────────────────────────
st.header("2. SMOTE — Balancing the Dataset")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Before SMOTE (training set)**")
    before = smote_counts["before"]
    st.metric("Legitimate", f"{int(before.get(0.0, before.get('0', 0))):,}")
    st.metric("Fraud",      f"{int(before.get(1.0, before.get('1', 0))):,}")
with c2:
    st.markdown("**After SMOTE (training set)**")
    after = smote_counts["after"]
    st.metric("Legitimate", f"{int(after.get(0.0, after.get('0', 0))):,}")
    st.metric("Fraud",      f"{int(after.get(1.0, after.get('1', 0))):,}")

fig_smote, ax = plt.subplots(figsize=(7, 3.5))
b_vals = [int(before.get(0.0, before.get('0', 0))),
          int(before.get(1.0, before.get('1', 0)))]
a_vals = [int(after.get(0.0,  after.get('0',  0))),
          int(after.get(1.0,  after.get('1',  0)))]
x = np.arange(2)
w = 0.35
ax.bar(x - w/2, b_vals, w, label="Before SMOTE", color="#e74c3c", alpha=0.75, edgecolor="black")
ax.bar(x + w/2, a_vals, w, label="After SMOTE",  color="#2ecc71", alpha=0.75, edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
ax.set_ylabel("Samples"); ax.set_title("SMOTE Impact on Class Balance")
ax.legend()
st.pyplot(fig_smote)
plt.close(fig_smote)

# ── Model comparison ───────────────────────────────────────────────────────────
st.header("3. Model Comparison")

def style_recall(val):
    color = "#2ecc71" if val >= 0.8 else ("#f39c12" if val >= 0.5 else "#e74c3c")
    return f"background-color: {color}22; font-weight: bold;"

styled = (comparison.style
          .format({c: "{:.4f}" for c in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]})
          .map(style_recall, subset=["Recall"]))
st.dataframe(styled, use_container_width=True)

best_row = comparison.iloc[0]
st.success(f"🏆 **Best model (highest Recall):** {best_row['Model']}  — "
           f"Recall = {best_row['Recall']:.4f} | F1 = {best_row['F1-Score']:.4f} | ROC-AUC = {best_row['ROC-AUC']:.4f}")

# ── Detailed charts ────────────────────────────────────────────────────────────
st.header("4. Detailed Visualizations")
tab1, tab2, tab3, tab4 = st.tabs(["Metrics", "Confusion Matrices", "ROC Curves", "Feature Importance"])

with tab1:
    fig_metrics, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#95a5a6", "#2ecc71", "#3498db", "#f39c12"]
    metrics = ["Recall", "F1-Score", "ROC-AUC"]
    for i, metric in enumerate(metrics):
        names = comparison["Model"].tolist()
        metric_vals = comparison[metric].tolist()
        axes[i].barh(names, metric_vals, color=colors, edgecolor="black", alpha=0.8)
        axes[i].set_xlabel(metric); axes[i].set_title(f"Model Comparison — {metric}")
        for j, v in enumerate(metric_vals):
            axes[i].text(v + 0.005, j, f"{v:.3f}", va="center", fontsize=9)
        axes[i].set_xlim(0, 1.1)
    fig_metrics.tight_layout()
    st.pyplot(fig_metrics)
    plt.close(fig_metrics)

with tab2:
    fig_cm, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, preds) in zip(axes, models_preds.items()):
        cm = confusion_matrix(y_test, preds["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with tab3:
    fig_roc, ax = plt.subplots(figsize=(8, 6))
    colors_roc = ["#95a5a6", "#2ecc71", "#3498db", "#f39c12"]
    for color, (name, preds) in zip(colors_roc, models_preds.items()):
        fpr, tpr, _ = roc_curve(y_test, preds["y_proba"])
        auc = roc_auc_score(y_test, preds["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2, color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    st.pyplot(fig_roc)
    plt.close(fig_roc)

with tab4:
    fig_fi, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_imp["Feature"], feature_imp["Importance"],
            color="lightgreen", edgecolor="black")
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Feature Importances (Random Forest + SMOTE)")
    ax.invert_yaxis()
    fig_fi.tight_layout()
    st.pyplot(fig_fi)
    plt.close(fig_fi)

# ── Live prediction ────────────────────────────────────────────────────────────
st.header("5. Live Transaction Prediction")
st.markdown("Enter a transaction's features below, or use the **random** buttons to generate a sample.")

rf_model  = results["rf_model"]
scaler    = results["scaler"]
X_columns = results["X_columns"]

col_gen1, col_gen2, _ = st.columns([1, 1, 4])
gen_legit = col_gen1.button("Generate legitimate sample")
gen_fraud = col_gen2.button("Generate suspicious sample")

if gen_legit:
    np.random.seed(int(np.random.randint(0, 9999)))
    st.session_state["pred_input"] = {
        **{f"V{i}": float(np.random.randn() * 0.5) for i in range(1, 29)},
        "Amount": float(np.random.uniform(1, 300)),
    }
if gen_fraud:
    np.random.seed(int(np.random.randint(0, 9999)))
    st.session_state["pred_input"] = {
        **{f"V{i}": float(np.random.randn() * 2 + 1) for i in range(1, 29)},
        "Amount": float(np.random.uniform(2000, 5000)),
    }

defaults = st.session_state.get("pred_input", {})

with st.form("prediction_form"):
    n_cols = 6
    cols = st.columns(n_cols)
    inputs = {}
    for idx, col_name in enumerate(X_columns):
        default_val = defaults.get(col_name, 0.0)
        inputs[col_name] = cols[idx % n_cols].number_input(
            col_name, value=round(default_val, 4), format="%.4f", step=0.0001,
            key=f"inp_{col_name}"
        )
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df)
    pred = rf_model.predict(input_scaled)[0]
    prob = rf_model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"🚨 **FRAUD DETECTED** — Fraud probability: {prob:.2%}")
    else:
        st.success(f"✅ **LEGITIMATE** — Fraud probability: {prob:.2%}")

    fig_gauge, ax = plt.subplots(figsize=(5, 2.5))
    bar_color = "#e74c3c" if pred == 1 else "#2ecc71"
    ax.barh(["Fraud probability"], [prob], color=bar_color, edgecolor="black")
    ax.barh(["Fraud probability"], [1 - prob], left=[prob], color="#ecf0f1", edgecolor="black")
    ax.set_xlim(0, 1); ax.set_xlabel("Probability")
    ax.set_title(f"Fraud Probability: {prob:.2%}")
    st.pyplot(fig_gauge)
    plt.close(fig_gauge)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Credit Card Fraud Detection | SMOTE Imbalanced Learning Demo | March 2026")