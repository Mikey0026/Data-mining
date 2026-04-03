from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Phishing Prototype", page_icon="🛡️", layout="wide")

DATA_PATH = Path(__file__).with_name("Phishing_Websites_Data.csv")

FEATURE_HELP = {
    "having_IP_Address": "IP-based URLs are often more suspicious than normal domain names.",
    "URL_Length": "Very long URLs can hide phishing tricks.",
    "Shortining_Service": "Shortened URLs can hide the real destination.",
    "having_At_Symbol": "An @ symbol inside a URL is a common phishing warning sign.",
    "double_slash_redirecting": "Unexpected redirect patterns may indicate phishing.",
    "Prefix_Suffix": "Hyphenated domains are often riskier.",
    "SSLfinal_State": "Poor or missing SSL/TLS is a risk signal.",
    "Request_URL": "Heavy use of external resources can be suspicious.",
    "URL_of_Anchor": "Suspicious anchor behavior may indicate phishing.",
    "SFH": "Server form handler issues are a known phishing indicator.",
    "web_traffic": "Lower traffic can indicate a less trustworthy site.",
    "Google_Index": "Sites not indexed by Google can be riskier.",
}

PRIMARY_FEATURES = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "web_traffic",
    "Google_Index",
    "Statistical_report",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("ï»¿", "").replace("\ufeff", "") for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = clean_columns(df)

    target_col = next(col for col in df.columns if str(col).lower() == "result")
    y = pd.to_numeric(df[target_col], errors="coerce").replace({-1: 0, 1: 1})

    valid_mask = y.notna()
    df = df.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    X = df.drop(columns=[c for c in [target_col, "URL", "url"] if c in df.columns], errors="ignore").copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))
    return X, y


@st.cache_resource(show_spinner=False)
def train_model():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = HistGradientBoostingClassifier(
        random_state=42,
        max_depth=8,
        learning_rate=0.08,
        max_iter=200,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "feature_columns": list(X.columns),
        "defaults": X.mode().iloc[0].to_dict(),
    }
    return model, metrics


def format_name(name: str) -> str:
    return name.replace("_", " ")


def risk_band(prob: float) -> str:
    if prob >= 0.8:
        return "High risk"
    if prob >= 0.5:
        return "Medium risk"
    return "Low risk"


@st.cache_data(show_spinner=False)
def get_demo_examples():
    X, y = load_dataset()
    examples = {
        "Likely phishing website": X.loc[y == 1].head(1).iloc[0].to_dict(),
        "Likely legitimate website": X.loc[y == 0].head(1).iloc[0].to_dict(),
        "Typical mixed website": X.median(numeric_only=True).round().astype(int).to_dict(),
    }
    return examples


def build_input_frame(values, feature_columns, defaults):
    prepared = {col: values.get(col, defaults.get(col, 0)) for col in feature_columns}
    return pd.DataFrame([prepared]).reindex(columns=feature_columns, fill_value=0)


def show_prediction_result(input_df: pd.DataFrame, probability: float, source_label: str):
    prediction = int(probability >= 0.5)
    label = "Phishing" if prediction == 1 else "Legitimate"
    risk = risk_band(probability)

    st.subheader(f"{source_label} result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted class", label)
    c2.metric("Phishing probability", f"{probability:.1%}")
    c3.metric("Risk level", risk)

    if prediction == 1:
        st.error("This website pattern looks suspicious and is likely phishing.")
    elif probability >= 0.35:
        st.warning("This website looks mostly legitimate, but a few signals are still worth checking.")
    else:
        st.success("This website pattern looks more legitimate in this demo.")

    preview_cols = [col for col in PRIMARY_FEATURES if col in input_df.columns][:8]
    if preview_cols:
        preview_df = pd.DataFrame(
            {
                "Feature": [format_name(col) for col in preview_cols],
                "Value": [int(input_df.iloc[0][col]) for col in preview_cols],
            }
        )
        st.caption("Feature snapshot used for this prediction")
        st.dataframe(preview_df, use_container_width=True, hide_index=True)


try:
    X_demo, _ = load_dataset()
    model, metrics = train_model()
except Exception as exc:
    st.title("🛡️ Phishing Website Detector")
    st.error(f"The app could not start because of a data-loading problem: {exc}")
    st.stop()

feature_columns = metrics["feature_columns"]
defaults = metrics["defaults"]
demo_examples = get_demo_examples()
sample_batch = X_demo.head(5).reset_index(drop=True).reindex(columns=feature_columns)

st.title("🛡️ Phishing Website Detector")
st.write(
    "This class demo uses a machine-learning model trained on `Phishing_Websites_Data.csv` to "
    "estimate whether a website pattern looks **phishing** or **legitimate**."
)
st.info(
    "Class demo tip: start with the **Quick demo** tab, then show either the manual input or CSV upload."
)

m1, m2, m3 = st.columns(3)
m1.metric("Holdout accuracy", f"{metrics['accuracy']:.3f}")
m2.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.3f}")
m3.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

tab1, tab2, tab3 = st.tabs(["Quick demo", "Manual input", "Batch CSV"])

with tab1:
    st.markdown("### Ready-made examples")
    st.write("Choose a preloaded example to show the prototype instantly in class.")

    demo_choice = st.selectbox("Select an example", list(demo_examples.keys()))
    if st.button("Run selected example", type="primary", use_container_width=True):
        demo_input_df = build_input_frame(demo_examples[demo_choice], feature_columns, defaults)
        demo_probability = float(model.predict_proba(demo_input_df)[0, 1])
        show_prediction_result(demo_input_df, demo_probability, demo_choice)

    st.download_button(
        "Download example CSV for batch scoring",
        sample_batch.to_csv(index=False).encode("utf-8"),
        file_name="prototype_demo_examples.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("This file matches the model's expected feature columns and can be uploaded in the Batch CSV tab.")

with tab2:
    st.markdown("### Manual phishing check")
    st.caption("Value guide: `-1` = lower-risk pattern, `0` = neutral/unknown, `1` = higher-risk pattern in this demo dataset.")

    manual_values = {}
    shown_primary = [col for col in PRIMARY_FEATURES if col in feature_columns]
    remaining_features = [col for col in feature_columns if col not in shown_primary]

    left_col, right_col = st.columns(2)
    split_index = (len(shown_primary) + 1) // 2

    for container, features in ((left_col, shown_primary[:split_index]), (right_col, shown_primary[split_index:])):
        with container:
            for col in features:
                dataset_options = sorted({int(v) for v in X_demo[col].dropna().unique().tolist()})
                options = dataset_options if dataset_options else [-1, 0, 1]
                default_val = int(defaults.get(col, options[0])) if pd.notna(defaults.get(col, options[0])) else options[0]
                index = options.index(default_val) if default_val in options else 0
                manual_values[col] = st.selectbox(
                    format_name(col),
                    options=options,
                    index=index,
                    help=FEATURE_HELP.get(col),
                    key=f"manual_{col}",
                )

    with st.expander("Additional technical features"):
        for col in remaining_features:
            dataset_options = sorted({int(v) for v in X_demo[col].dropna().unique().tolist()})
            options = dataset_options if dataset_options else [-1, 0, 1]
            default_val = int(defaults.get(col, options[0])) if pd.notna(defaults.get(col, options[0])) else options[0]
            index = options.index(default_val) if default_val in options else 0
            manual_values[col] = st.selectbox(
                format_name(col),
                options=options,
                index=index,
                key=f"extra_{col}",
            )

    if st.button("Predict manual values", use_container_width=True):
        input_df = build_input_frame(manual_values, feature_columns, defaults)
        probability = float(model.predict_proba(input_df)[0, 1])
        show_prediction_result(input_df, probability, "Manual input")

with tab3:
    st.markdown("### Batch scoring")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with matching feature columns to score many rows at once",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_df = clean_columns(batch_df)
            original_batch = batch_df.copy()

            batch_features = batch_df.drop(
                columns=[c for c in ["Result", "result", "URL", "url"] if c in batch_df.columns],
                errors="ignore",
            ).copy()
            for col in batch_features.columns:
                batch_features[col] = pd.to_numeric(batch_features[col], errors="coerce")

            batch_features = batch_features.replace([np.inf, -np.inf], np.nan)
            missing_cols = [col for col in feature_columns if col not in batch_features.columns]
            batch_features = batch_features.reindex(columns=feature_columns)
            batch_features = batch_features.fillna(pd.Series(defaults))

            if missing_cols:
                preview_missing = ", ".join(missing_cols[:8])
                suffix = " ..." if len(missing_cols) > 8 else ""
                st.warning(
                    f"{len(missing_cols)} missing feature columns were filled automatically with default values: {preview_missing}{suffix}"
                )

            batch_prob = model.predict_proba(batch_features)[:, 1]
            batch_pred = (batch_prob >= 0.5).astype(int)

            output_df = original_batch.copy()
            output_df["predicted_class"] = np.where(batch_pred == 1, "phishing", "legitimate")
            output_df["phishing_probability"] = batch_prob

            st.write("Preview of scored rows")
            st.dataframe(output_df.head(20), use_container_width=True)
            st.download_button(
                "Download scored CSV",
                output_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_phishing_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Could not score the uploaded file: {exc}")

st.caption(
    "Note: this is a demonstration prototype for academic use. Real-world phishing detection should include stronger "
    "feature engineering, monitoring, and additional security checks."
)
#working prototype is located at http://localhost:8502