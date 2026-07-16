from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="Skin Disease Multimodal Classifier",
    layout="wide",
)


ARTIFACT_DIR = Path("saved_model_artifacts")
FALLBACK_ARTIFACT_DIR = Path(".")


def get_artifact_path(filename: str) -> Path:
    """Find artifacts in saved_model_artifacts first, then in the project root."""
    artifact_path = ARTIFACT_DIR / filename
    if artifact_path.exists():
        return artifact_path

    fallback_path = FALLBACK_ARTIFACT_DIR / filename
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"Could not find '{filename}' in '{ARTIFACT_DIR}' or the project root."
    )


def load_pickle(filename: str):
    """Load one pickle artifact by filename."""
    with get_artifact_path(filename).open("rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner="Loading model and preprocessing artifacts...")
def load_artifacts():
    """Load the trained Keras model and all preprocessing objects once."""
    model = tf.keras.models.load_model(
        get_artifact_path("multimodal_skin_model.keras"),
        compile=False,
    )

    artifacts = {
        "model": model,
        "label_encoder": load_pickle("label_encoder.pkl"),
        "sex_encoder": load_pickle("sex_encoder.pkl"),
        "localization_encoder": load_pickle("localization_encoder.pkl"),
        "age_scaler": load_pickle("age_scaler.pkl"),
        "disease_name_map": load_pickle("disease_name_map.pkl"),
        "app_config": load_pickle("app_config.pkl"),
    }
    return artifacts


def normalize_options(options):
    """Convert config options to clean display strings for Streamlit dropdowns."""
    return [str(option) for option in list(options)]


def preprocess_image(uploaded_file, img_size: int) -> np.ndarray:
    """Resize, normalize, and batch the uploaded skin lesion image."""
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((img_size, img_size))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def preprocess_tabular(age, sex, localization, artifacts) -> np.ndarray:
    """Scale and encode tabular inputs in the model's training order."""
    age_scaled = artifacts["age_scaler"].transform([[age]])[0][0]
    sex_encoded = artifacts["sex_encoder"].transform([sex])[0]
    localization_encoded = artifacts["localization_encoder"].transform([localization])[0]

    return np.array(
        [[age_scaled, sex_encoded, localization_encoded]],
        dtype=np.float32,
    )


def predict(image_array: np.ndarray, tabular_array: np.ndarray, artifacts):
    """Run model inference and decode the top class."""
    probabilities = artifacts["model"].predict(
        [image_array, tabular_array],
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_code = artifacts["label_encoder"].inverse_transform([predicted_index])[0]
    predicted_name = artifacts["disease_name_map"].get(
        predicted_code,
        str(predicted_code).upper(),
    )
    confidence = float(probabilities[predicted_index])

    return predicted_code, predicted_name, confidence, probabilities


def build_probability_table(probabilities: np.ndarray, artifacts) -> pd.DataFrame:
    """Create a sorted table with class codes, full names, and probabilities."""
    class_codes = artifacts["label_encoder"].inverse_transform(
        np.arange(len(probabilities))
    )

    probability_rows = []
    for code, probability in zip(class_codes, probabilities):
        probability_rows.append(
            {
                "Disease": artifacts["disease_name_map"].get(code, str(code).upper()),
                "Code": code,
                "Probability": float(probability),
            }
        )

    return pd.DataFrame(probability_rows).sort_values(
        by="Probability",
        ascending=True,
    )


def plot_probabilities(probability_table: pd.DataFrame):
    """Render a horizontal probability bar chart."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#2563eb"] * len(probability_table)
    colors[-1] = "#16a34a"

    ax.barh(
        probability_table["Disease"],
        probability_table["Probability"] * 100,
        color=colors,
    )
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for index, probability in enumerate(probability_table["Probability"]):
        ax.text(
            min(probability * 100 + 1.5, 96),
            index,
            f"{probability * 100:.1f}%",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def plot_probability_donut(probability_table: pd.DataFrame):
    """Render a compact donut chart for predicted class probabilities."""
    chart_table = probability_table.sort_values(by="Probability", ascending=False)
    labels = chart_table["Disease"]
    values = chart_table["Probability"]
    colors = [
        "#16a34a",
        "#2563eb",
        "#f59e0b",
        "#dc2626",
        "#7c3aed",
        "#0891b2",
        "#64748b",
    ]

    fig, ax = plt.subplots(figsize=(6, 4.6))
    wedges, _ = ax.pie(
        values,
        startangle=90,
        colors=colors[: len(values)],
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
    )
    ax.text(
        0,
        0,
        "Class\nprobability",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#111827",
    )
    ax.legend(
        wedges,
        labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        fontsize=9,
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_class_overview(artifacts):
    """Show the seven supported disease classes before prediction."""
    classes = artifacts["app_config"].get("classes")
    if classes is None:
        classes = artifacts["label_encoder"].classes_

    class_names = [
        artifacts["disease_name_map"].get(class_code, str(class_code).upper())
        for class_code in classes
    ]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#64748b"]
    ax.bar(class_names, [1] * len(class_names), color=colors[: len(class_names)])
    ax.set_title("Supported Disease Classes", fontsize=14, fontweight="bold")
    ax.set_ylabel("Model output class")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    fig.tight_layout()
    return fig


def confidence_badge(confidence: float) -> tuple[str, str]:
    """Return label and color based on confidence."""
    if confidence >= 0.75:
        return "High confidence", "#15803d"
    if confidence >= 0.45:
        return "Moderate confidence", "#d97706"
    return "Low confidence", "#f59e0b"


def display_results(uploaded_file, predicted_name, confidence, probabilities, artifacts):
    """Show the uploaded image, prediction, confidence, and class probabilities."""
    badge_text, badge_color = confidence_badge(confidence)
    probability_table = build_probability_table(probabilities, artifacts)

    st.subheader("Prediction Result")

    image_col, result_col = st.columns([1, 1.25], gap="large")
    with image_col:
        uploaded_file.seek(0)
        st.image(uploaded_file, caption="Uploaded lesion image", use_container_width=True)

    with result_col:
        st.markdown(
            f"""
            <div style="padding: 1.2rem; border-left: 6px solid {badge_color};
                        background: #f8fafc; border-radius: 0.5rem;">
                <div style="font-size: 0.9rem; color: {badge_color};
                            font-weight: 700; text-transform: uppercase;">
                    {badge_text}
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: #111827;
                            margin-top: 0.35rem;">
                    {predicted_name}
                </div>
                <div style="font-size: 1.2rem; color: #374151; margin-top: 0.35rem;">
                    Confidence: <strong>{confidence * 100:.2f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    chart_col, donut_col = st.columns([1.25, 0.85], gap="large")
    with chart_col:
        st.markdown("#### Class Probability Breakdown")
        st.pyplot(plot_probabilities(probability_table))

    with donut_col:
        st.markdown("#### Probability Share")
        st.pyplot(plot_probability_donut(probability_table))

    with st.expander("View probability table"):
        table = probability_table.sort_values(by="Probability", ascending=False).copy()
        table["Probability"] = table["Probability"].map(lambda value: f"{value * 100:.2f}%")
        st.dataframe(table, use_container_width=True, hide_index=True)


def main():
    st.title("Skin Disease Multimodal Classifier")
    st.write(
        "Upload a skin lesion image and enter patient information to predict one "
        "of seven HAM10000 skin disease classes using a trained multimodal model."
    )
    st.warning(
        "Educational and portfolio use only. This app is not a medical device and "
        "must not be used for real diagnosis or treatment decisions."
    )

    try:
        artifacts = load_artifacts()
    except Exception as error:
        st.error("The model artifacts could not be loaded.")
        st.exception(error)
        return

    app_config = artifacts["app_config"]
    img_size = int(app_config.get("img_size", 224))
    sex_options = normalize_options(app_config.get("sex_options", []))
    localization_options = normalize_options(app_config.get("localization_options", []))

    input_col, output_col = st.columns([0.9, 1.4], gap="large")

    with input_col:
        st.subheader("Patient Inputs")
        uploaded_file = st.file_uploader(
            "Skin lesion photo",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear JPG or PNG image of the lesion.",
        )
        age = st.slider("Age", min_value=0, max_value=100, value=50, step=1)
        sex = st.selectbox("Sex", options=sex_options)
        localization = st.selectbox("Body localization", options=localization_options)
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    with output_col:
        if not predict_clicked:
            st.info("Upload an image, enter patient details, and click Predict.")
            st.pyplot(plot_class_overview(artifacts))
            return

        if uploaded_file is None:
            st.warning("Please upload a skin lesion image before predicting.")
            st.pyplot(plot_class_overview(artifacts))
            return

        try:
            with st.spinner("Preprocessing inputs and running the model..."):
                image_array = preprocess_image(uploaded_file, img_size)
                tabular_array = preprocess_tabular(
                    age,
                    sex,
                    localization,
                    artifacts,
                )
                _, predicted_name, confidence, probabilities = predict(
                    image_array,
                    tabular_array,
                    artifacts,
                )

            display_results(
                uploaded_file,
                predicted_name,
                confidence,
                probabilities,
                artifacts,
            )
        except ValueError as error:
            st.error(
                "One of the selected patient values is not recognized by the saved "
                "encoders. Please choose values from the dropdown lists."
            )
            st.exception(error)
        except Exception as error:
            st.error("Prediction failed. Please check the uploaded image and artifacts.")
            st.exception(error)


if __name__ == "__main__":
    main()
