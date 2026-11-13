from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit


# -----------------------------------------------------------------------------
# App settings and dataset information
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "jakarta_traffic.csv.gz"

PERIODS = {
    "Night (00:00–06:00)": "night",
    "Morning peak (06:00–09:00)": "morning_peak",
    "Morning off-peak (09:00–12:00)": "morning_offpeak",
    "Lunch hours (12:00–14:00)": "lunch_hours",
    "Afternoon off-peak (14:00–16:00)": "afternoon_offpeak",
    "Evening peak (16:00–19:00)": "evening_peak",
    "Evening off-peak (19:00–22:00)": "evening_offpeak",
    "Late night (22:00–00:00)": "late_night",
}
PERIOD_ORDER = list(PERIODS.values())
PERIOD_TO_CODE = {name: code for code, name in enumerate(PERIOD_ORDER)}
FEATURES = ["longitude", "latitude", "road_length_km", "period_code"]
TARGET = "jam_factor_mean"

st.set_page_config(
    page_title="Jakarta Traffic Predictor",
    page_icon="🚦",
    layout="wide",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
      div[data-testid="stMetric"] {
        background: #f6f8fb; border: 1px solid #e5eaf1;
        padding: 16px 18px; border-radius: 14px;
      }
      div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        border-radius: 10px; font-weight: 650; min-height: 2.8rem;
      }
      .small-note { color: #667085; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Load the small, preprocessed Jakarta road-segment dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Make sure the data folder is included in the repository."
        )
    data = pd.read_csv(DATA_PATH, compression="gzip")
    required = {"fid", "longitude", "latitude", "road_length_km", TARGET, "period"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    data["period_code"] = data["period"].map(PERIOD_TO_CODE)
    if data["period_code"].isna().any():
        raise ValueError("The dataset contains an unknown time period.")
    return data


@st.cache_resource
def train_model():
    """Evaluate on unseen road segments, then train the final model on all rows."""
    data = load_data()
    x = data[FEATURES].astype(float)
    y = data[TARGET].astype(float)
    groups = data["fid"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))

    validation_model = HistGradientBoostingRegressor(
        max_iter=120,
        max_leaf_nodes=20,
        learning_rate=0.08,
        l2_regularization=1.0,
        random_state=42,
    )
    validation_model.fit(x.iloc[train_idx], y.iloc[train_idx])
    validation_predictions = validation_model.predict(x.iloc[test_idx])
    validation_mae = mean_absolute_error(y.iloc[test_idx], validation_predictions)

    final_model = HistGradientBoostingRegressor(
        max_iter=120,
        max_leaf_nodes=20,
        learning_rate=0.08,
        l2_regularization=1.0,
        random_state=42,
    )
    final_model.fit(x, y)
    return final_model, float(validation_mae)


def haversine_km(lat, lon, latitudes, longitudes):
    """Return approximate great-circle distances from one point to many points."""
    earth_radius_km = 6371.0
    lat1 = np.radians(float(lat))
    lon1 = np.radians(float(lon))
    lat2 = np.radians(np.asarray(latitudes, dtype=float))
    lon2 = np.radians(np.asarray(longitudes, dtype=float))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def traffic_label(jam_factor):
    """Convert the dataset's 0–10 jam factor to its documented category."""
    if jam_factor < 1.0:
        return "Free flow"
    if jam_factor < 3.0:
        return "Light traffic"
    if jam_factor < 6.0:
        return "Moderate traffic"
    if jam_factor < 8.0:
        return "Heavy traffic"
    return "Severe congestion"


# -----------------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------------
try:
    data = load_data()
except Exception as exc:  # Show a useful message instead of a long app traceback.
    st.error(f"Could not load the traffic dataset: {exc}")
    st.stop()

st.title("🚦 Jakarta Traffic Predictor")
st.write(
    "Estimate the typical congestion level for a road location in Jakarta "
    "based on its coordinates and the time of day."
)
st.caption("Historical baseline · March 2025 to February 2026 · 14,549 road segments")

with st.expander("What does this app predict?", expanded=False):
    st.write(
        "The model estimates the **average jam factor** for a road location and time period. "
        "Jam factor is a congestion score from 0 to 10; it is not a count of cars or a travel-time estimate."
    )
    st.write(
        "The uploaded dataset does not contain vehicle counts, weather, incidents, or live traffic. "
        "For that reason, this version uses location, the selected time period, and the length of the "
        "nearest mapped road segment."
    )

with st.sidebar:
    st.header("Prediction inputs")
    st.write("Enter a point within the Jakarta area covered by the dataset.")

    lat_min = float(data["latitude"].min())
    lat_max = float(data["latitude"].max())
    lon_min = float(data["longitude"].min())
    lon_max = float(data["longitude"].max())

    with st.form("prediction_form"):
        latitude = st.number_input(
            "Latitude",
            min_value=lat_min,
            max_value=lat_max,
            value=float(data["latitude"].median()),
            step=0.001,
            format="%.6f",
            help="Use the latitude coordinate of your selected location.",
        )
        longitude = st.number_input(
            "Longitude",
            min_value=lon_min,
            max_value=lon_max,
            value=float(data["longitude"].median()),
            step=0.001,
            format="%.6f",
            help="Use the longitude coordinate of your selected location.",
        )
        selected_period_label = st.selectbox("Time of day", list(PERIODS.keys()), index=5)
        predict_clicked = st.form_submit_button("Predict traffic", type="primary", use_container_width=True)

    st.caption(
        f"Dataset coordinate range: {lat_min:.3f} to {lat_max:.3f} latitude, "
        f"{lon_min:.3f} to {lon_max:.3f} longitude."
    )

st.subheader("Your prediction")
if predict_clicked:
    with st.spinner("Preparing the model and estimating congestion…"):
        model, validation_mae = train_model()
        period = PERIODS[selected_period_label]

        # Find the nearest road segment; its length is used as a model input.
        distances = haversine_km(
            latitude,
            longitude,
            data["latitude"].to_numpy(),
            data["longitude"].to_numpy(),
        )
        nearest_index = int(np.argmin(distances))
        nearest_segment = data.iloc[nearest_index]

        model_input = pd.DataFrame(
            [{
                "longitude": float(longitude),
                "latitude": float(latitude),
                "road_length_km": float(nearest_segment["road_length_km"]),
                "period_code": float(PERIOD_TO_CODE[period]),
            }],
            columns=FEATURES,
        )
        prediction = float(np.clip(model.predict(model_input)[0], 0.0, 10.0))
        category = traffic_label(prediction)

        metric_cols = st.columns(3)
        metric_cols[0].metric("Predicted jam factor", f"{prediction:.2f} / 10")
        metric_cols[1].metric("Traffic category", category)
        metric_cols[2].metric("Nearest road segment", f"#{int(nearest_segment['fid'])}")

        st.progress(min(max(prediction / 10.0, 0.0), 1.0), text=f"Congestion score: {prediction:.2f} out of 10")
        st.caption(
            f"Selected period: **{selected_period_label}** · "
            f"Nearest mapped segment: **{distances[nearest_index]:.2f} km** away · "
            f"Its mapped road length is **{nearest_segment['road_length_km']:.2f} km**."
        )

        # Show the five nearest observed segments for context.
        same_period = data.loc[data["period"] == period].copy()
        same_period_distances = haversine_km(
            latitude,
            longitude,
            same_period["latitude"].to_numpy(),
            same_period["longitude"].to_numpy(),
        )
        same_period["distance_km"] = same_period_distances
        nearby = same_period.nsmallest(5, "distance_km").copy()
        nearby["traffic_level"] = nearby[TARGET].map(traffic_label)

        left, right = st.columns([1.15, 0.85], gap="large")
        with left:
            st.markdown("#### Nearby road segments")
            st.write("Recorded historical averages for the selected time period:")
            display_nearby = nearby[["fid", "distance_km", TARGET, "traffic_level"]].rename(
                columns={
                    "fid": "Segment ID",
                    "distance_km": "Distance (km)",
                    TARGET: "Recorded jam factor",
                    "traffic_level": "Traffic level",
                }
            )
            st.dataframe(
                display_nearby.style.format(
                    {"Distance (km)": "{:.3f}", "Recorded jam factor": "{:.2f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.markdown("#### Congestion scale")
            st.write("0–10 jam-factor interpretation from the dataset documentation:")
            st.write("- **0–1:** Free flow")
            st.write("- **1–3:** Light traffic")
            st.write("- **3–6:** Moderate traffic")
            st.write("- **6–8:** Heavy traffic")
            st.write("- **8–10:** Severe congestion")

        st.metric("Validation MAE", f"{validation_mae:.3f} jam-factor points")
        st.caption(
            "Validation uses a held-out 20% of road segments, so the same road segment does not appear "
            "in both the training and validation groups. This score measures error on this historical dataset; "
            "it does not guarantee real-world or live-traffic accuracy."
        )

        with st.expander("View nearby points on a map"):
            map_points = nearby[["latitude", "longitude"]].copy()
            selected_point = pd.DataFrame({"latitude": [latitude], "longitude": [longitude]})
            st.map(pd.concat([map_points, selected_point], ignore_index=True), zoom=11)
            st.caption("Dots show nearby road-segment centroids and your selected point.")
else:
    st.info("Choose a location and time period in the sidebar, then select **Predict traffic**.")

st.divider()

with st.expander("Average congestion by time of day"):
    period_summary = (
        data.groupby("period", as_index=False)[TARGET]
        .mean()
        .assign(period=lambda frame: pd.Categorical(frame["period"], categories=PERIOD_ORDER, ordered=True))
        .sort_values("period")
        .rename(columns={TARGET: "Average jam factor"})
        .set_index("period")
    )
    st.bar_chart(period_summary)

with st.expander("About the dataset and model"):
    st.markdown(
        """
        - **Source:** HERE Traffic API road-segment observations and OpenStreetMap road geometry.
        - **Coverage:** Jakarta, 14,549 road segments, aggregated into eight daily time periods.
        - **Period covered:** March 2025–February 2026.
        - **Target:** `jam_factor_mean`, the mean jam factor for each road segment and time period.
        - **Model:** `HistGradientBoostingRegressor` using latitude, longitude, nearest road-segment length, and time period.
        - **Important limitation:** The files contain no vehicle-count variable. `jam_factor_count` in the original files is the number of 15-minute observations contributing to an aggregate, **not** the number of cars.
        """
    )

st.caption(
    "Educational prototype only. This app describes historical congestion patterns and is not a live navigation service."
)
