import datetime
import os
import altair as alt
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Plume — Air Quality Forecasting",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(html_str: str):
    """Strips leading whitespace per line so Markdown never renders HTML as a code block."""
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Loading & Caching
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("models/best_aqi_model.joblib")
    meta = joblib.load("models/metadata.joblib")
    return model, meta


@st.cache_data
def load_historical_lookup(cities, global_medians):
    csv_path = "data/city_day.csv"
    pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    city_stats = {}
    city_month_profiles = {}

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.month

        city_med_df = df.groupby("City")[pollutant_cols].median()
        city_month_df = df.groupby(["City", "Month"])[pollutant_cols].median()

        for city_name, grp in df.groupby("City"):
            aqi_clean = grp["AQI"].dropna()
            if len(aqi_clean) > 0:
                city_stats[city_name] = {
                    "median": float(round(aqi_clean.median(), 1)),
                    "mean": float(round(aqi_clean.mean(), 1)),
                    "max": float(round(aqi_clean.max(), 1)),
                    "min": float(round(aqi_clean.min(), 1)),
                }

        for city in cities:
            for month in range(1, 13):
                vals = {}
                for col in pollutant_cols:
                    v = None
                    if (city, month) in city_month_df.index:
                        val = city_month_df.loc[(city, month), col]
                        if pd.notna(val):
                            v = float(val)
                    if v is None and city in city_med_df.index:
                        val = city_med_df.loc[city, col]
                        if pd.notna(val):
                            v = float(val)
                    if v is None:
                        v = float(global_medians[col])
                    vals[col] = v
                city_month_profiles[(city, month)] = vals

    return city_stats, city_month_profiles


try:
    model, meta = load_models()
    city_stats, city_month_profiles = load_historical_lookup(meta["cities"], meta["medians"])
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()


def get_baseline_for_city_month(city: str, month: int):
    if (city, month) in city_month_profiles:
        return city_month_profiles[(city, month)]
    return meta["medians"]


SLIDER_BOUNDS = {
    "pm25": 1000.0,
    "pm10": 1200.0,
    "no2": 500.0,
    "so2": 500.0,
    "co": 50.0,
    "o3": 500.0,
}


def set_baseline_to_state(city: str, month: int):
    baseline = get_baseline_for_city_month(city, month)
    st.session_state["pm25"] = min(SLIDER_BOUNDS["pm25"], max(0.0, float(round(baseline["PM2.5"], 1))))
    st.session_state["pm10"] = min(SLIDER_BOUNDS["pm10"], max(0.0, float(round(baseline["PM10"], 1))))
    st.session_state["no2"] = min(SLIDER_BOUNDS["no2"], max(0.0, float(round(baseline["NO2"], 1))))
    st.session_state["so2"] = min(SLIDER_BOUNDS["so2"], max(0.0, float(round(baseline["SO2"], 1))))
    st.session_state["co"] = min(SLIDER_BOUNDS["co"], max(0.0, float(round(baseline["CO"], 2))))
    st.session_state["o3"] = min(SLIDER_BOUNDS["o3"], max(0.0, float(round(baseline["O3"], 1))))


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    render_html(
        """
        <div style="padding-bottom: 1rem; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.08);">
            <div style="font-size: 1.35rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em;">Plume</div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.15rem;">India Air Quality Forecasting</div>
        </div>
        """
    )

    cities = meta["cities"]
    qp = st.query_params
    qp_city = qp.get("city", "Delhi")
    default_idx = cities.index(qp_city) if qp_city in cities else (cities.index("Delhi") if "Delhi" in cities else 0)
    default_date = datetime.date.today()
    if "date" in qp:
        try:
            default_date = datetime.date.fromisoformat(qp["date"])
        except ValueError:
            pass
    selected_city = st.selectbox("City", cities, index=default_idx)
    selected_date = st.date_input("Date", default_date)

    # Auto-load historical city + month medians whenever city or month changes
    current_ctx = (selected_city, selected_date.month)
    if st.session_state.get("last_city_month") != current_ctx:
        set_baseline_to_state(selected_city, selected_date.month)
        st.session_state["last_city_month"] = current_ctx

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    render_html(
        "<div style='font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:#94A3B8; margin-bottom:0.5rem;'>Pollutant Concentrations</div>"
    )

    pm25 = st.slider("PM2.5 (µg/m³)", 0.0, SLIDER_BOUNDS["pm25"], step=1.0, key="pm25")
    pm10 = st.slider("PM10 (µg/m³)", 0.0, SLIDER_BOUNDS["pm10"], step=1.0, key="pm10")
    no2 = st.slider("NO2 (µg/m³)", 0.0, SLIDER_BOUNDS["no2"], step=1.0, key="no2")
    so2 = st.slider("SO2 (µg/m³)", 0.0, SLIDER_BOUNDS["so2"], step=1.0, key="so2")
    co = st.slider("CO (mg/m³)", 0.0, SLIDER_BOUNDS["co"], step=0.05, key="co")
    o3 = st.slider("O3 (µg/m³)", 0.0, SLIDER_BOUNDS["o3"], step=1.0, key="o3")

    st.button(
        "Reset to city typical",
        on_click=set_baseline_to_state,
        args=(selected_city, selected_date.month),
        width="stretch",
    )

# ---------------------------------------------------------
# Machine Learning Prediction
# ---------------------------------------------------------
input_dict = {col: 0.0 for col in meta["feature_columns"]}
input_dict.update({
    "Year": float(selected_date.year),
    "Month": float(selected_date.month),
    "DayOfWeek": float(selected_date.weekday()),
    "PM2.5": float(pm25),
    "PM10": float(pm10),
    "NO2": float(no2),
    "SO2": float(so2),
    "CO": float(co),
    "O3": float(o3),
})

city_col = f"City_{selected_city}"
if city_col in input_dict:
    input_dict[city_col] = 1.0

input_df = pd.DataFrame([input_dict])[meta["feature_columns"]]
predicted_aqi = max(0.0, float(model.predict(input_df)[0]))

# Dominant Pollutant relative to CPCB safe limits
cpcb_limits = {"PM2.5": 60.0, "PM10": 100.0, "NO2": 80.0, "SO2": 80.0, "CO": 2.0, "O3": 100.0}
ratios = {
    "PM2.5": pm25 / cpcb_limits["PM2.5"],
    "PM10": pm10 / cpcb_limits["PM10"],
    "NO2": no2 / cpcb_limits["NO2"],
    "SO2": so2 / cpcb_limits["SO2"],
    "CO": co / cpcb_limits["CO"],
    "O3": o3 / cpcb_limits["O3"],
}
dominant_pollutant = max(ratios, key=ratios.get)
dominant_mult = ratios[dominant_pollutant]

# Particulate ratio
pm_ratio = (pm25 / pm10) if pm10 > 0 else 0.0
pm_source = "Combustion-dominated" if pm_ratio > 0.5 else "Dust / crustal source"

# ---------------------------------------------------------
# Automatic Atmospheric Theme (Season + Predicted AQI)
# ---------------------------------------------------------
month = selected_date.month
is_monsoon = month in (6, 7, 8, 9)       # Jun - Sep
is_winter = month in (11, 12, 1, 2)      # Nov - Feb

if predicted_aqi > 250:
    active_theme = "smog"
elif predicted_aqi > 150:
    active_theme = "winter" if (is_winter or month == 10) else "haze"
elif is_monsoon and predicted_aqi <= 150:
    active_theme = "rain"
elif is_winter and predicted_aqi > 90:
    active_theme = "winter"
else:
    active_theme = "sunny"

ATMOSPHERE_STYLES = {
    "rain": {
        "bg": "radial-gradient(circle at 85% 15%, rgba(56, 189, 248, 0.25) 0%, transparent 45%), linear-gradient(160deg, #0B2545 0%, #133C55 50%, #1B4965 100%)",
        "condition": "Monsoon Rain & Washout",
    },
    "sunny": {
        "bg": "radial-gradient(circle at 85% 12%, rgba(253, 224, 71, 0.22) 0%, transparent 40%), linear-gradient(160deg, #0369A1 0%, #0284C7 55%, #0EA5E9 100%)",
        "condition": "Clear & Sunny",
    },
    "haze": {
        "bg": "radial-gradient(circle at 80% 15%, rgba(251, 191, 36, 0.16) 0%, transparent 45%), linear-gradient(160deg, #1E293B 0%, #334155 55%, #475569 100%)",
        "condition": "Warm Urban Haze",
    },
    "winter": {
        "bg": "radial-gradient(circle at 80% 15%, rgba(148, 163, 184, 0.18) 0%, transparent 45%), linear-gradient(160deg, #1E293B 0%, #273447 55%, #334155 100%)",
        "condition": "Cool Winter Mist",
    },
    "smog": {
        "bg": "radial-gradient(circle at 80% 15%, rgba(234, 88, 12, 0.18) 0%, transparent 48%), linear-gradient(160deg, #181312 0%, #2C1D18 50%, #3D241C 100%)",
        "condition": "Severe Smog & Thermal Inversion" if is_winter else "High Pollution Smog",
    },
}

cur_atmo = ATMOSPHERE_STYLES[active_theme]

# CPCB AQI Category & Health Guidance
if predicted_aqi <= 50:
    aqi_cat, aqi_col, aqi_tint = "Good", "#22C55E", "rgba(34, 197, 94, 0.16)"
    summary = "Air quality is clean and poses minimal health risk."
    mask_adv = "Not required"
    ex_adv = "Ideal for outdoor sports"
    vent_adv = "Open windows for fresh air"
elif predicted_aqi <= 100:
    aqi_cat, aqi_col, aqi_tint = "Satisfactory", "#84CC16", "rgba(132, 204, 22, 0.16)"
    summary = "Acceptable air quality; minor discomfort possible for sensitive groups."
    mask_adv = "Optional"
    ex_adv = "Safe for normal workouts"
    vent_adv = "Normal natural ventilation"
elif predicted_aqi <= 200:
    aqi_cat, aqi_col, aqi_tint = "Moderate", "#EAB308", "rgba(234, 179, 8, 0.16)"
    summary = "Breathing discomfort to people with lung or heart conditions, children, and seniors."
    mask_adv = "Recommended for sensitive groups"
    ex_adv = "Limit prolonged outdoor cardio"
    vent_adv = "Ventilate in afternoon hours"
elif predicted_aqi <= 300:
    aqi_cat, aqi_col, aqi_tint = "Poor", "#F97316", "rgba(249, 115, 22, 0.16)"
    summary = "Breathing discomfort to most people on prolonged exposure."
    mask_adv = "N95 mask advised outdoors"
    ex_adv = "Avoid strenuous outdoor workouts"
    vent_adv = "Keep windows closed; run HEPA"
elif predicted_aqi <= 400:
    aqi_cat, aqi_col, aqi_tint = "Very Poor", "#EF4444", "rgba(239, 68, 68, 0.16)"
    summary = "Respiratory illness on prolonged exposure; avoid outdoor exertion."
    mask_adv = "N95 respirator required outdoors"
    ex_adv = "Cancel all outdoor activities"
    vent_adv = "Seal windows; run air purifiers"
else:
    aqi_cat, aqi_col, aqi_tint = "Severe", "#A855F7", "rgba(168, 85, 247, 0.18)"
    summary = "Affects healthy adults and seriously impacts those with respiratory or cardiac diseases."
    mask_adv = "N95 respirator mandatory"
    ex_adv = "Remain strictly indoors"
    vent_adv = "Run HEPA purifiers at max"

needle_pct = min(100.0, max(0.0, (predicted_aqi / 500.0) * 100.0))

# ---------------------------------------------------------
# Designer CSS (Full-Viewport Atmospheric Sky + Frosted Glass Bento Cards)
# ---------------------------------------------------------
render_html(
    f"""
    <style>
        html, body, .stApp {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
            background: transparent !important;
            color: #F8FAFC !important;
        }}

        [data-testid="stAppViewContainer"] {{
            background: {cur_atmo['bg']} !important;
            background-attachment: fixed !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        [data-testid="stSidebar"] {{
            background-color: #0E1219 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }}

        #MainMenu, footer {{
            visibility: hidden;
        }}

        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1120px !important;
        }}

        /* Top Location Header */
        .top-bar {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        }}

        .city-title {{
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.05;
            color: #FFFFFF;
            margin: 0;
        }}

        .city-meta {{
            font-size: 0.92rem;
            color: rgba(255, 255, 255, 0.85);
            margin-top: 0.35rem;
            font-weight: 500;
        }}

        .brand-pill {{
            font-size: 0.75rem;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.95);
            background: rgba(12, 17, 29, 0.56);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.16);
            padding: 0.4rem 0.85rem;
            border-radius: 999px;
            letter-spacing: 0.02em;
        }}

        /* Frosted Glass Bento Cards */
        .glass-card {{
            background: rgba(12, 17, 29, 0.58);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 16px;
            padding: 1.5rem;
            min-height: 335px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
        }}

        .st-key-chart_card, [data-testid="stExpander"] {{
            background: rgba(12, 17, 29, 0.58) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(255, 255, 255, 0.14) !important;
            border-radius: 16px !important;
        }}

        .st-key-chart_card {{
            padding: 1.25rem 1.5rem 0.75rem 1.5rem !important;
            margin-top: 1rem !important;
        }}

        [data-testid="stExpander"] details summary {{
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }}

        .card-eyebrow {{
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: rgba(255, 255, 255, 0.6);
        }}

        .aqi-top-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: {aqi_tint};
            color: {aqi_col};
            border: 1px solid {aqi_col}44;
        }}

        .status-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: {aqi_col};
            display: inline-block;
        }}

        .aqi-number-row {{
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            margin: 0.25rem 0 0.75rem 0;
        }}

        .aqi-score {{
            font-size: 4.4rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.04em;
            color: #FFFFFF;
            font-variant-numeric: tabular-nums;
        }}

        .aqi-unit {{
            font-size: 0.95rem;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.55);
            letter-spacing: 0.04em;
        }}

        .aqi-summary {{
            font-size: 0.88rem;
            color: rgba(255, 255, 255, 0.85);
            line-height: 1.45;
            margin-bottom: 1.35rem;
        }}

        /* Thin Refined Scale Bar */
        .scale-bar {{
            height: 6px;
            border-radius: 999px;
            background: linear-gradient(
                to right,
                #22C55E 0%, #22C55E 10%,
                #84CC16 10%, #84CC16 20%,
                #EAB308 20%, #EAB308 40%,
                #F97316 40%, #F97316 60%,
                #EF4444 60%, #EF4444 80%,
                #A855F7 80%, #A855F7 100%
            );
            position: relative;
        }}

        .scale-dot {{
            position: absolute;
            top: -5px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #FFFFFF;
            border: 3px solid #0F172A;
            transform: translateX(-50%);
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.5);
        }}

        .scale-ticks {{
            display: flex;
            justify-content: space-between;
            font-size: 0.68rem;
            color: rgba(255, 255, 255, 0.5);
            margin-top: 0.45rem;
            font-variant-numeric: tabular-nums;
        }}

        /* 2x2 Guidance Subgrid */
        .guidance-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
            margin-top: 1rem;
        }}

        .guidance-tile {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 0.85rem 1rem;
        }}

        .guidance-label {{
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: rgba(255, 255, 255, 0.55);
            margin-bottom: 0.25rem;
        }}

        .guidance-val {{
            font-size: 0.88rem;
            font-weight: 600;
            color: #FFFFFF;
            line-height: 1.35;
        }}

        .guidance-sub {{
            font-size: 0.74rem;
            color: rgba(255, 255, 255, 0.6);
            margin-top: 0.15rem;
        }}

        .meta-footer-row {{
            display: flex;
            justify-content: space-between;
            margin-top: 1rem;
            padding-top: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 0.76rem;
            color: rgba(255, 255, 255, 0.65);
        }}
    </style>
    """
)

# ---------------------------------------------------------
# Top Location Header
# ---------------------------------------------------------
render_html(
    f"""
    <div class="top-bar">
        <div>
            <h1 class="city-title">{selected_city}</h1>
            <div class="city-meta">{selected_date.strftime('%A, %d %b %Y')} · {cur_atmo['condition']}</div>
        </div>
        <div class="brand-pill">Plume · CPCB Standard</div>
    </div>
    """
)

# ---------------------------------------------------------
# Bento Row 1: Hero AQI Card & Guidance Card
# ---------------------------------------------------------
col_left, col_right = st.columns([1.0, 1.05], gap="medium")

with col_left:
    render_html(
        f"""
        <div class="glass-card">
            <div class="aqi-top-row">
                <span class="card-eyebrow">Air Quality Index</span>
                <span class="status-pill">
                    <span class="status-dot"></span>
                    <span>{aqi_cat}</span>
                </span>
            </div>

            <div class="aqi-number-row">
                <span class="aqi-score">{predicted_aqi:.0f}</span>
                <span class="aqi-unit">AQI</span>
            </div>

            <div class="aqi-summary">{summary}</div>

            <div class="scale-bar">
                <div class="scale-dot" style="left: {needle_pct:.1f}%;"></div>
            </div>
            <div class="scale-ticks">
                <span>0</span>
                <span>50</span>
                <span>100</span>
                <span>200</span>
                <span>300</span>
                <span>400</span>
                <span>500+</span>
            </div>
        </div>
        """
    )

with col_right:
    city_info = city_stats.get(selected_city)
    hist_med_str = f"{city_info['median']:.0f} AQI" if city_info else "—"
    render_html(
        f"""
        <div class="glass-card">
            <div class="card-eyebrow">Conditions & Health Guidance</div>

            <div class="guidance-grid">
                <div class="guidance-tile">
                    <div class="guidance-label">Primary Driver</div>
                    <div class="guidance-val">{dominant_pollutant}</div>
                    <div class="guidance-sub">{dominant_mult:.1f}x safe standard</div>
                </div>
                <div class="guidance-tile">
                    <div class="guidance-label">Mask</div>
                    <div class="guidance-val">{mask_adv}</div>
                </div>
                <div class="guidance-tile">
                    <div class="guidance-label">Outdoor Activity</div>
                    <div class="guidance-val">{ex_adv}</div>
                </div>
                <div class="guidance-tile">
                    <div class="guidance-label">Ventilation</div>
                    <div class="guidance-val">{vent_adv}</div>
                </div>
            </div>

            <div class="meta-footer-row">
                <span>PM2.5/PM10: <strong>{pm_ratio:.2f}</strong> ({pm_source})</span>
                <span>{selected_city} 5-yr median: <strong>{hist_med_str}</strong></span>
            </div>
        </div>
        """
    )

# ---------------------------------------------------------
# Bento Row 2: Integrated Pollutant Concentrations Card
# ---------------------------------------------------------
with st.container(border=True, key="chart_card"):
    render_html(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <span class="card-eyebrow" style="font-size: 0.76rem; color: rgba(255,255,255,0.75);">Pollutant Concentrations <span style="text-transform: none;">(µg/m³)</span></span>
            <span style="font-size: 0.72rem; color: rgba(255,255,255,0.55);">
                <span style="color:#38BDF8;">●</span> Within limit &nbsp;&nbsp;
                <span style="color:#F87171;">●</span> Exceeds CPCB standard
            </span>
        </div>
        """
    )

    chart_df = pd.DataFrame([
        {"Pollutant": "PM2.5", "Level": float(pm25), "Standard": 60.0, "Exceeded": pm25 > 60.0},
        {"Pollutant": "PM10", "Level": float(pm10), "Standard": 100.0, "Exceeded": pm10 > 100.0},
        {"Pollutant": "NO2", "Level": float(no2), "Standard": 80.0, "Exceeded": no2 > 80.0},
        {"Pollutant": "SO2", "Level": float(so2), "Standard": 80.0, "Exceeded": so2 > 80.0},
        {"Pollutant": "CO (×10)", "Level": float(co * 10.0), "Standard": 20.0, "Exceeded": co > 2.0},
        {"Pollutant": "O3", "Level": float(o3), "Standard": 100.0, "Exceeded": o3 > 100.0},
    ])

    y_max = max(50.0, float(chart_df["Level"].max() * 1.18 + 10.0))

    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, width=44)
        .encode(
            x=alt.X("Pollutant:N", sort=None, axis=alt.Axis(labelColor="#E2E8F0", title=None, labelFontSize=12, labelFontWeight="bold", labelAngle=0)),
            y=alt.Y("Level:Q", scale=alt.Scale(domain=[0, y_max]), axis=alt.Axis(labelColor="#94A3B8", title=None, gridColor="rgba(255,255,255,0.06)")),
            color=alt.condition(
                alt.datum.Exceeded,
                alt.value("#F87171"),
                alt.value("#38BDF8"),
            ),
            tooltip=[
                alt.Tooltip("Pollutant:N"),
                alt.Tooltip("Level:Q", format=".1f", title="Level"),
                alt.Tooltip("Standard:Q", format=".1f", title="CPCB Standard"),
            ],
        )
    )

    text_labels = bars.mark_text(
        align="center",
        baseline="bottom",
        dy=-6,
        color="#F8FAFC",
        fontSize=12,
        fontWeight="bold",
    ).encode(
        text=alt.Text("Level:Q", format=".1f")
    )

    pollutant_chart = (
        (bars + text_labels)
        .properties(height=230, background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
        .configure_axis(
            labelColor="#E2E8F0",
            gridColor="rgba(255, 255, 255, 0.06)",
            domainColor="rgba(255, 255, 255, 0.12)",
        )
    )

    st.altair_chart(pollutant_chart, width="stretch")

# ---------------------------------------------------------
# Collapsible Historical Context & Model Info
# ---------------------------------------------------------
render_html("<div style='height: 0.75rem;'></div>")
with st.expander(f"Historical Baseline for {selected_city} & Model Details", expanded=False):
    c_data = city_stats.get(selected_city)
    if c_data:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("5-Year Median AQI", f"{c_data['median']:.0f}")
        m2.metric("5-Year Average AQI", f"{c_data['mean']:.0f}")
        m3.metric("Cleanest Recorded Day", f"{c_data['min']:.0f}")
        m4.metric("Worst Recorded Day", f"{c_data['max']:.0f}")
    st.caption("Plume · Random Forest Regressor (R² 0.908, MAE 21.1 AQI) trained on 2015–2020 CPCB India data.")
