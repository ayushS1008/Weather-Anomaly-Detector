# Weather Anomaly Detector

A compact Streamlit dashboard for monitoring automatic weather stations. It demonstrates robust z-score anomaly detection for temperature, humidity, and wind readings.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The included data is deterministic demo telemetry. Replace `make_readings()` in `app.py` with your station feed when you are ready to connect live data.
