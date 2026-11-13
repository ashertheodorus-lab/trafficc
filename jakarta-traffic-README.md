# Jakarta Traffic Predictor

A small Streamlit web app that estimates the **historical average congestion level** (jam factor, 0–10) for a location in Jakarta and a selected time of day.

The layout is inspired by the simple single-app Streamlit structure in the example repository, but the model and data pipeline are built for this Jakarta traffic dataset.

## What the app uses

- Latitude and longitude entered by the user
- Selected time period (one of eight periods)
- Length of the nearest mapped road segment, retrieved from the dataset

The model is a `HistGradientBoostingRegressor`. It is evaluated with a 20% holdout split grouped by road-segment ID, then trained on all rows for the app's predictions.

## Important dataset limitation

The uploaded dataset **does not include the number of cars or vehicle volume**. The original `jam_factor_count` column counts the number of 15-minute observations contributing to an aggregate; it is **not** a vehicle count. Therefore, this app predicts jam factor from location and time period, not from the amount of cars.

To make a model that predicts congestion from the number of cars, the dataset would need actual vehicle-count/traffic-volume measurements aligned to the same road segments and time periods. Weather, incidents, and day-of-week data could also improve a future version.

## Dataset details

- Jakarta road network: 14,549 road segments
- Time coverage: March 2025 to February 2026
- Eight daily periods: night, morning peak, morning off-peak, lunch hours, afternoon off-peak, evening peak, evening off-peak, and late night
- Target: `jam_factor_mean`, from 0 to 10
- Source described in the uploaded `DATA_README.md`: HERE Traffic API observations and OpenStreetMap road geometry
- The original GeoPackage files have been converted into a compact CSV gzip file for easy deployment.

The congestion categories documented with the dataset are:

| Jam factor | Interpretation |
|---:|---|
| 0–1 | Free flow |
| 1–3 | Light traffic |
| 3–6 | Moderate traffic |
| 6–8 | Heavy traffic |
| 8–10 | Severe congestion |

## Run locally

1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

Streamlit will print a local address (normally `http://localhost:8501`) to open in your browser.

## Publish with GitHub and Streamlit Community Cloud

1. Sign in to GitHub and create a **new public repository** (for example, `jakarta-traffic-predictor`).
2. Upload the contents of this project folder. Keep `app.py`, `requirements.txt`, the `.streamlit` folder, and the `data` folder in the repository root structure shown here.
3. Sign in to Streamlit Community Cloud and choose **Create app**.
4. Connect the GitHub repository, choose the branch (normally `main`), and set the main file path to `app.py`.
5. Deploy the app. Streamlit installs the packages listed in `requirements.txt` and runs `app.py`.

## Project structure

```text
jakarta_traffic_predictor/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
    └── jakarta_traffic.csv.gz
```

## Notes

This is an educational prototype using historical aggregates. It is not connected to a live traffic feed, and its output should not be treated as a real-time traffic forecast or navigation instruction.
