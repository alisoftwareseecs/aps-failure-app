import streamlit as st
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import xgboost as xgb

st.set_page_config(page_title="APS Failure Diagnosis", layout="wide")
COST_FP, COST_FN = 10, 500

@st.cache_resource
def load_and_train():
    train = pd.read_csv("aps_failure_training_set.csv", skiprows=20, na_values=["na"])
    test = pd.read_csv("aps_failure_test_set.csv", skiprows=20, na_values=["na"])
    y_full = (train["class"] == "pos").astype(int)
    y_test = (test["class"] == "pos").astype(int)
    X_full, X_test = train.drop(columns=["class"]).copy(), test.drop(columns=["class"]).copy()
    X_full["n_missing"] = X_full.isna().sum(axis=1)
    X_test["n_missing"] = X_test.isna().sum(axis=1)
    drop_cols = X_full.drop(columns=["n_missing"]).isna().mean().pipe(lambda s: s[s > 0.7].index.tolist())
    X_full, X_test = X_full.drop(columns=drop_cols), X_test.drop(columns=drop_cols)
    cols = list(X_full.columns)
    X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
    imp = SimpleImputer(strategy="median").fit(X_tr)
    X_tr_i = pd.DataFrame(imp.transform(X_tr), columns=cols)
    X_val_i = pd.DataFrame(imp.transform(X_val), columns=cols)
    X_test_i = pd.DataFrame(imp.transform(X_test), columns=cols)
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    model = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
                               random_state=42, n_jobs=4).fit(X_tr_i, y_tr)
    val_proba = model.predict_proba(X_val_i)[:, 1]
    test_proba = model.predict_proba(X_test_i)[:, 1]
    return model, imp, cols, drop_cols, y_val, val_proba, y_test, test_proba, test

def cost_at(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
    return COST_FP * fp + COST_FN * fn, fp, fn, tp, tn

model, imputer, cols, drop_cols, y_val, val_proba, y_test, test_proba, test_raw = load_and_train()
best_t = min(np.linspace(0.001, 0.5, 300), key=lambda t: cost_at(y_val, val_proba, t)[0])

st.title("🚛 APS Failure Diagnosis — Scania Trucks")
st.caption("Predicts whether a failed truck's root cause is the Air Pressure System (APS) or another subsystem.")

thresh = st.sidebar.slider("Decision threshold", 0.0, 1.0, float(round(best_t, 3)), 0.005)
st.sidebar.write(f"Cost-optimal threshold (from validation sweep): **{best_t:.3f}**")
st.sidebar.markdown(f"Cost matrix: FP = {COST_FP}, FN = {COST_FN}")

cost, fp, fn, tp, tn = cost_at(y_test, test_proba, thresh)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test ROC-AUC", f"{roc_auc_score(y_test, test_proba):.3f}")
c2.metric("Test PR-AUC", f"{average_precision_score(y_test, test_proba):.3f}")
c3.metric("Total cost @ threshold", f"{cost:,.0f}")
c4.metric("Recall (pos)", f"{tp/(tp+fn):.2%}" if (tp+fn) else "n/a")

st.subheader("Confusion matrix on held-out test set")
cm_df = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual neg", "Actual pos"], columns=["Pred neg", "Pred pos"])
st.dataframe(cm_df, use_container_width=True)

st.divider()
st.subheader("Try a truck from the test set")
idx = st.number_input("Row index (0–%d)" % (len(test_raw) - 1), 0, len(test_raw) - 1, 0)
row = test_raw.drop(columns=["class"]).iloc[[idx]].copy()
row["n_missing"] = row.isna().sum(axis=1)
row = row.drop(columns=[c for c in drop_cols if c in row.columns])[cols]
row_imp = pd.DataFrame(imputer.transform(row), columns=cols)
proba = model.predict_proba(row_imp)[0, 1]
actual = test_raw.iloc[idx]["class"]
pred = "pos (APS)" if proba >= thresh else "neg (other)"
st.write(f"**Predicted probability of APS failure:** {proba:.3f}")
st.write(f"**Decision @ threshold {thresh:.3f}:** {pred}  |  **Actual label:** {actual}")

st.divider()
st.subheader("Upload your own batch (must match raw column schema)")
up = st.file_uploader("CSV with the same raw columns as the training set", type="csv")
if up:
    batch = pd.read_csv(up, na_values=["na"])
    has_label = "class" in batch.columns
    X = batch.drop(columns=["class"]) if has_label else batch.copy()
    X["n_missing"] = X.isna().sum(axis=1)
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    X = X.reindex(columns=cols)
    X_imp = pd.DataFrame(imputer.transform(X), columns=cols)
    p = model.predict_proba(X_imp)[:, 1]
    out = batch.copy()
    out["aps_probability"] = p
    out["prediction"] = np.where(p >= thresh, "pos (APS)", "neg (other)")
    st.dataframe(out, use_container_width=True)
    if has_label:
        yb = (batch["class"] == "pos").astype(int)
        cb, fpb, fnb, tpb, tnb = cost_at(yb, p, thresh)
        st.write(f"Batch total cost @ threshold {thresh:.3f}: **{cb:,.0f}**  (FP={fpb}, FN={fnb})")
