import os
from datetime import date
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

DEFAULT_API_BASE_URL = "http://regression-mlops-e2e-alb-1975146041.us-east-1.elb.amazonaws.com"

REQUIRED_FIELDS = ["date", "city_full", "city", "zipcode"]
MARKET_FIELDS = [
    "median_sale_price",
    "median_list_price",
    "median_ppsf",
    "median_list_ppsf",
    "homes_sold",
    "pending_sales",
    "new_listings",
    "inventory",
    "median_dom",
    "avg_sale_to_list",
    "sold_above_list",
    "off_market_in_two_weeks",
]
POI_FIELDS = [
    "bank",
    "bus",
    "hospital",
    "mall",
    "park",
    "restaurant",
    "school",
    "station",
    "supermarket",
]
DEMOGRAPHIC_FIELDS = [
    "total_population",
    "median_age",
    "per_capita_income",
    "total_families_below_poverty",
    "total_housing_units",
    "median_rent",
    "median_home_value",
    "total_labor_force",
    "unemployed_population",
    "total_school_age_population",
    "total_school_enrollment",
    "median_commute_time",
]
ALL_OPTIONAL_FIELDS = MARKET_FIELDS + POI_FIELDS + DEMOGRAPHIC_FIELDS

SAMPLE_RECORD = {
    "date": "2022-01-01",
    "city_full": "Atlanta-Sandy Springs-Alpharetta",
    "city": "ATL",
    "zipcode": 30301,
    "median_sale_price": 395000,
    "median_list_price": 420000,
    "median_ppsf": 215,
    "median_list_ppsf": 225,
    "homes_sold": 620,
    "pending_sales": 510,
    "new_listings": 780,
    "inventory": 1600,
    "median_dom": 28,
    "avg_sale_to_list": 0.99,
    "sold_above_list": 0.31,
    "off_market_in_two_weeks": 0.42,
    "bank": 12,
    "bus": 35,
    "hospital": 4,
    "mall": 2,
    "park": 18,
    "restaurant": 75,
    "school": 21,
    "station": 6,
    "supermarket": 14,
    "total_population": 498715,
    "median_age": 34.8,
    "per_capita_income": 48200,
    "total_families_below_poverty": 11800,
    "total_housing_units": 235000,
    "median_rent": 1850,
    "median_home_value": 390000,
    "total_labor_force": 276000,
    "unemployed_population": 9400,
    "total_school_age_population": 80500,
    "total_school_enrollment": 72100,
    "median_commute_time": 27,
}


def configure_page() -> None:
    st.set_page_config(
        page_title="Regression MLOps Prediction Console",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem;}
        div[data-testid="stMetric"] {border: 1px solid #e5e7eb; padding: 0.75rem 0.9rem; border-radius: 8px;}
        div[data-testid="stStatusWidget"] {visibility: hidden; height: 0; position: fixed;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_api_base(value: str) -> str:
    value = value.strip().rstrip("/")
    if value.endswith("/predict"):
        return value.removesuffix("/predict")
    if value.endswith("/predict/batch"):
        return value.removesuffix("/predict/batch")
    return value


@st.cache_data(ttl=30)
def get_json(base_url: str, path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(f"{base_url}{path}", timeout=10)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int = 45) -> dict[str, Any]:
    response = requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def money(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"${float(value):,.0f}"


def compact_label(name: str) -> str:
    return name.replace("_", " ").title()


def optional_number_input(field: str, default: float | int | None = None) -> float | None:
    enabled = st.checkbox(compact_label(field), value=default is not None, key=f"enabled_{field}")
    if not enabled:
        return None
    value = float(default or 0)
    return st.number_input("Value", value=value, key=f"value_{field}", label_visibility="collapsed")


def build_single_payload() -> dict[str, Any]:
    use_sample = st.toggle("Use sample values", value=True)
    source = SAMPLE_RECORD if use_sample else {}

    col1, col2, col3, col4 = st.columns([1.1, 2.2, 1.1, 1])
    with col1:
        raw_date = st.date_input("Date", value=date.fromisoformat(source.get("date", "2022-01-01")))
    with col2:
        city_full = st.text_input("Metro area", value=source.get("city_full", ""))
    with col3:
        city = st.text_input("City code", value=source.get("city", ""))
    with col4:
        zipcode = st.number_input("Zipcode", min_value=0, value=int(source.get("zipcode", 0)), step=1)

    payload: dict[str, Any] = {
        "date": raw_date.isoformat(),
        "city_full": city_full,
        "city": city,
        "zipcode": int(zipcode),
    }

    tabs = st.tabs(["Market", "Places", "Demographics"])
    groups = [MARKET_FIELDS, POI_FIELDS, DEMOGRAPHIC_FIELDS]
    for tab, fields in zip(tabs, groups, strict=True):
        with tab:
            cols = st.columns(3)
            for index, field in enumerate(fields):
                with cols[index % 3]:
                    value = optional_number_input(field, source.get(field) if use_sample else None)
                    if value is not None:
                        payload[field] = value

    return payload


def render_prediction(result: dict[str, Any]) -> None:
    prediction = result.get("predicted_price")
    missing = result.get("missing_features", [])
    model_version = result.get("model_version", "-")

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Predicted price", money(prediction))
    c2.metric("Missing optional fields", len(missing))
    c3.metric("Model version", model_version.split("@")[0] if model_version else "-")

    if missing:
        st.dataframe(pd.DataFrame({"missing_optional_feature": missing}), use_container_width=True, hide_index=True)


def render_batch(base_url: str) -> None:
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        sample = pd.DataFrame([SAMPLE_RECORD, {**SAMPLE_RECORD, "zipcode": 30303, "median_list_price": 455000}])
        st.download_button(
            "Download sample CSV",
            data=sample.to_csv(index=False),
            file_name="prediction_sample.csv",
            mime="text/csv",
        )
        return

    df = pd.read_csv(uploaded)
    st.dataframe(df.head(20), use_container_width=True)

    missing_required = [field for field in REQUIRED_FIELDS if field not in df.columns]
    if missing_required:
        st.error(f"Missing required columns: {', '.join(missing_required)}")
        return

    max_rows = st.slider("Rows to score", min_value=1, max_value=min(len(df), 1000), value=min(len(df), 50))
    records = df.head(max_rows).where(pd.notnull(df.head(max_rows)), None).to_dict(orient="records")

    if st.button("Run batch prediction", type="primary", use_container_width=True):
        try:
            result = post_json(base_url, "/predict/batch", {"records": records}, timeout=90)
        except requests.RequestException as exc:
            st.error(f"Batch request failed: {exc}")
            return

        predictions = result.get("predictions", [])
        out = df.head(max_rows).copy()
        out["predicted_price"] = [item.get("predicted_price") for item in predictions]
        out["missing_optional_count"] = [len(item.get("missing_features", [])) for item in predictions]

        st.metric("Rows predicted", result.get("rows_predicted", len(out)))
        st.dataframe(out, use_container_width=True)

        if "price" in out.columns:
            scored = out.dropna(subset=["price", "predicted_price"]).copy()
            if not scored.empty:
                scored["absolute_error"] = (scored["predicted_price"] - scored["price"]).abs()
                mae = scored["absolute_error"].mean()
                rmse = ((scored["predicted_price"] - scored["price"]) ** 2).mean() ** 0.5
                m1, m2 = st.columns(2)
                m1.metric("MAE", money(mae))
                m2.metric("RMSE", money(rmse))

                chart_df = scored.reset_index().rename(columns={"index": "row"})
                fig = px.line(chart_df, x="row", y=["price", "predicted_price"], markers=True)
                fig.update_layout(height=360, legend_title_text="Series")
                st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download predictions",
            data=out.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
        )


def main() -> None:
    configure_page()

    env_url = os.getenv("API_BASE_URL") or os.getenv("API_URL") or DEFAULT_API_BASE_URL
    with st.sidebar:
        st.header("Connection")
        api_base_url = normalize_api_base(st.text_input("API base URL", value=env_url))
        refresh = st.button("Refresh status", use_container_width=True)
        if refresh:
            get_json.clear()

        health, health_error = get_json(api_base_url, "/health")
        if health_error:
            st.error("API unavailable")
            st.caption(health_error)
        elif health:
            st.success(health.get("status", "unknown"))
            st.caption(health.get("model_version") or "model unavailable")

        model_info, _ = get_json(api_base_url, "/model-info") if health and health.get("model_loaded") else (None, None)
        if model_info:
            st.metric("Expected features", model_info.get("n_features_expected"))
            st.caption(model_info.get("source", ""))

    st.title("Regression MLOps Prediction Console")

    health, health_error = get_json(api_base_url, "/health")
    if health_error or not health or health.get("status") != "healthy":
        st.error("API health check failed")
        if health_error:
            st.code(health_error)
        return

    tab_single, tab_batch, tab_model = st.tabs(["Single prediction", "Batch CSV", "Model"])

    with tab_single:
        payload = build_single_payload()
        if st.button("Run prediction", type="primary", use_container_width=True):
            try:
                result = post_json(api_base_url, "/predict", payload)
            except requests.RequestException as exc:
                st.error(f"Prediction request failed: {exc}")
            else:
                render_prediction(result)
                with st.expander("Request payload"):
                    st.json(payload)

    with tab_batch:
        render_batch(api_base_url)

    with tab_model:
        model_info, model_error = get_json(api_base_url, "/model-info")
        if model_error:
            st.error(model_error)
        elif model_info:
            c1, c2, c3 = st.columns(3)
            c1.metric("Features", model_info.get("n_features_expected"))
            c2.metric("Source", model_info.get("source", "-").split("://")[0])
            c3.metric("Loaded at", str(model_info.get("loaded_at", "-"))[:19])
            st.code(model_info.get("version_string", ""))
            st.dataframe(pd.DataFrame({"train_columns": model_info.get("train_columns", [])}), use_container_width=True)


if __name__ == "__main__":
    main()
