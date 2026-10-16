"""
Hispanic Heritage Historical Data Explorer
Single-file Streamlit app: historical World Bank data, polynomial fitting,
scenario editing, exports, and classroom function/calculus exploration.
"""
from __future__ import annotations

import io
import math
import re
from datetime import date, datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current mortality patterns continued."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita (current US$)": ("NY.GDP.PCAP.CD", "current US dollars per person", "Gross domestic product divided by midyear population. This is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people (net count over period)", "Net number of migrants: immigrants minus emigrants. It is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people, as reported in the World Bank indicator."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a share of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, valued in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP; this is not the full current account."),
}
# Plausibility checks are deliberately broad screening rules, not claims that all
# values outside them are impossible.
BOUNDS = {
    "SP.DYN.LE00.IN": (0, 125), "SL.UEM.TOTL.NE.ZS": (0, 100),
    "NY.GDP.PCAP.CD": (0, None), "SP.DYN.CBRT.IN": (0, 100),
    "SM.POP.NETM": (None, None), "VC.IHR.PSRC.P5": (0, None),
    "FP.CPI.TOTL.ZG": (None, None), "GC.DOD.TOTL.GD.ZS": (0, None),
    "FI.RES.TOTL.CD": (0, None), "NE.RSB.GNFS.ZS": (None, None),
}
CONTEXTS = {
    "Business": ("Revenue", "thousand dollars", "time", "years", "p(x)=0.08*x**2 + 2*x + 10", "Revenue grows, and the derivative describes the rate of revenue change."),
    "Engineering": ("Position", "meters", "time", "seconds", "p(x)=x**2 - 4*x + 3", "The first derivative is velocity and the second derivative is acceleration."),
    "Medical": ("Concentration", "mg/L", "time", "hours", "p(x)=10*exp(-0.3*x)", "A concentration model can rise or decay; derivatives describe rate and change in rate."),
    "Sports science": ("Distance", "meters", "time", "seconds", "p(x)=2*x**2 + x", "Distance changes over time; derivative units are meters per second."),
    "Biology": ("Population", "organisms", "time", "days", "p(x)=100*exp(0.08*x)", "The first derivative is population growth rate."),
    "Bioengineering": ("Sensor output", "volts", "input", "seconds", "p(x)=sin(x)+2", "Derivatives describe how sensor output responds as input changes."),
    "Economic forecasting": ("GDP per capita", "current US dollars/person", "time", "years", "p(x)=0.5*x**2 + 100", "Rates describe modeled annual change, not a causal explanation."),
    "Weather": ("Temperature", "degrees C", "time", "hours", "p(x)=10 + 5*sin(2*pi*x/24)", "The first derivative describes warming/cooling rate."),
    "Custom": ("Output", "units", "input", "units", "p(x)=x**2", "Interpret the model only within the context and units you define."),
}
SAFE_LOCALS = {
    "x": sp.Symbol("x", real=True), "t": sp.Symbol("x", real=True),
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
    "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E,
}
X = SAFE_LOCALS["x"]


def wb_fetch(country_code: str, indicator_code: str, start: int, end: int) -> pd.DataFrame:
    """Fetch one indicator from the official World Bank API; no synthetic fallback."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {"date": f"{start}:{end}", "format": "json", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"World Bank request failed: {exc}") from exc
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise RuntimeError("The API returned an unexpected or empty response. Check the network or indicator code.")
    rows = []
    for item in payload[1]:
        if not item:
            continue
        try:
            year = int(item["date"])
            value = item.get("value")
            rows.append({"year": year, "value": float(value) if value is not None else np.nan})
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        raise RuntimeError("The API response contained no usable year/value rows.")
    frame = pd.DataFrame(rows).drop_duplicates("year").sort_values("year")
    frame = frame[(frame["year"] >= start) & (frame["year"] <= end)]
    return frame.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_wb_fetch(country_code: str, indicator_code: str, start: int, end: int) -> pd.DataFrame:
    return wb_fetch(country_code, indicator_code, start, end)


def validate_table(df: pd.DataFrame, indicator_code: str) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    if not {"year", "value"}.issubset(df.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must have year and value columns."]
    clean = df[["year", "value"]].copy()
    numeric_year = pd.to_numeric(clean["year"], errors="coerce")
    if numeric_year.isna().any() or ((numeric_year % 1) != 0).any():
        errors.append("Every year must be a whole number.")
    clean["year"] = numeric_year
    if clean["year"].isna().any():
        clean = clean.dropna(subset=["year"])
    clean["year"] = clean["year"].astype(int)
    if clean["year"].duplicated().any():
        errors.append("Years must be unique; remove duplicate years.")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    if clean["value"].isna().any():
        errors.append("Values must be numeric and not blank.")
    if not clean.empty and ((clean["year"] < 1900) | (clean["year"] > 2200)).any():
        errors.append("Years must be between 1900 and 2200.")
    low, high = BOUNDS[indicator_code]
    vals = clean["value"].dropna()
    if low is not None and (vals < low).any():
        errors.append(f"Values for this indicator should be at least {low}.")
    if high is not None and (vals > high).any():
        errors.append(f"Values for this indicator should be no more than {high}.")
    if clean["value"].notna().sum() < 2:
        errors.append("At least two numeric observations are needed.")
    return clean.sort_values("year").reset_index(drop=True), errors


def poly_fit(df: pd.DataFrame, degree: int):
    usable = df.dropna(subset=["year", "value"]).sort_values("year")
    if usable["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid years; current count is {usable['year'].nunique()}.")
    base = int(usable["year"].min())
    t = usable["year"].to_numpy(dtype=float) - base
    y = usable["value"].to_numpy(dtype=float)
    coeff = np.polynomial.polynomial.polyfit(t, y, degree)
    pred = np.polynomial.polynomial.polyval(t, coeff)
    r2 = float(r2_score(y, pred)) if len(y) > 1 else float("nan")
    mae = float(mean_absolute_error(y, pred))
    residuals = y - pred
    return {"base": base, "degree": degree, "coeff": coeff, "years": usable["year"].to_numpy(),
            "values": y, "fitted": pred, "residuals": residuals, "r2": r2, "mae": mae,
            "count": len(y), "min_year": int(usable["year"].min()), "max_year": int(usable["year"].max()),
            "mean": float(np.mean(y)), "frame": usable}


def poly_predict(model: dict, years):
    # Keep the fitted centered-time transformation. Do not center a second time.
    years = np.asarray(years, dtype=float)
    return np.polynomial.polynomial.polyval(years - model["base"], model["coeff"])


def equation_text(model: dict, units: str) -> str:
    parts = []
    for power, coef in enumerate(model["coeff"]):
        if abs(coef) < 1e-12:
            continue
        if power == 0:
            parts.append(f"{coef:.8g}")
        elif power == 1:
            parts.append(f"{coef:+.8g}t")
        else:
            parts.append(f"{coef:+.8g}t^{power}")
    return f"ŷ = {' '.join(parts) if parts else '0'}, where t = year − {model['base']} ({units} output)"


def safe_parse(expr: str):
    cleaned = expr.strip().replace("^", "**")
    if not cleaned:
        raise ValueError("Enter a function, for example x**2 + 3*x.")
    # Reject attribute access, brackets, quotes and characters outside the small grammar.
    if not re.fullmatch(r"[0-9a-zA-Z_+*/().,\-\s**]+", cleaned) or "__" in cleaned or "." in cleaned and ".." in cleaned:
        raise ValueError("Use ordinary math notation with numbers, x, +, -, *, /, powers, parentheses, and sin/cos/tan/exp/log/sqrt.")
    try:
        expr_obj = sp.sympify(cleaned, locals=SAFE_LOCALS, evaluate=True)
    except Exception as exc:
        raise ValueError(f"Could not parse that function: {exc}") from exc
    if expr_obj.free_symbols - {X}:
        raise ValueError("Only x is allowed as the independent variable.")
    return expr_obj


def numeric_function(expr, xs):
    fn = sp.lambdify(X, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        values = np.asarray(fn(xs), dtype=float)
    if values.ndim == 0:
        values = np.full(np.asarray(xs).shape, float(values))
    return values


def symbolic_analysis(expr, interval):
    a, b = interval
    d1, d2 = sp.diff(expr, X), sp.diff(expr, X, 2)
    try:
        critical_candidates = sp.solve(sp.Eq(d1, 0), X)
    except Exception:
        critical_candidates = []
    points = []
    for candidate in critical_candidates:
        if candidate.is_real is False or not candidate.is_real:
            continue
        try:
            val = float(candidate)
            if a <= val <= b and math.isfinite(val):
                left = float(d1.subs(X, val - 1e-4))
                right = float(d1.subs(X, val + 1e-4))
                kind = "local minimum" if left < 0 < right else "local maximum" if left > 0 > right else "stationary point (not classified as an extremum)"
                points.append((val, kind))
        except Exception:
            continue
    inflections = []
    try:
        candidates = sp.solve(sp.Eq(d2, 0), X)
    except Exception:
        candidates = []
    for candidate in candidates:
        if candidate.is_real is False or not candidate.is_real:
            continue
        try:
            val = float(candidate)
            if a < val < b and math.isfinite(val):
                left = float(d2.subs(X, val - 1e-4))
                right = float(d2.subs(X, val + 1e-4))
                if left * right < 0:
                    inflections.append(val)
        except Exception:
            pass
    # Absolute extrema are computed numerically with endpoints and derivative roots.
    candidates = [a, b] + [p[0] for p in points]
    valid = []
    for xx in candidates:
        try:
            yy = float(expr.subs(X, xx))
            if math.isfinite(yy):
                valid.append((xx, yy))
        except Exception:
            pass
    abs_min = min(valid, key=lambda z: z[1]) if valid else None
    abs_max = max(valid, key=lambda z: z[1]) if valid else None
    return d1, d2, points, inflections, abs_min, abs_max


def make_plot(frame, model, units, sample_step, single_year=None, single_value=None,
              show_derivatives=False, expr=None, point_x=None, tangent=False, secant=None):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(frame["year"], frame["value"], label="Observed / working data", s=22, zorder=4)
    hist_x = np.linspace(model["min_year"], model["max_year"], 600)
    ax.plot(hist_x, poly_predict(model, hist_x), label="Polynomial fit (smooth curve)", linewidth=2)
    sample_x = np.arange(model["min_year"], model["max_year"] + 1, sample_step)
    ax.scatter(sample_x, poly_predict(model, sample_x), label=f"Model sample markers (every {sample_step} year(s))", marker="x", s=28)
    if model.get("horizon", 0) > 0:
        future_x = np.linspace(model["max_year"], model["max_year"] + model["horizon"], 160)
        ax.plot(future_x, poly_predict(model, future_x), "--", label="Extrapolated segment")
    ax.axvline(model["max_year"], color="gray", linestyle=":", label="Historical boundary")
    if single_year is not None and single_value is not None:
        ax.scatter([single_year], [single_value], marker="*", s=130, label="Single-year estimate", zorder=5)
    if expr is not None and point_x is not None:
        lo, hi = secant if secant else (point_x - 1, point_x + 1)
        xs = np.linspace(lo, hi, 300)
        ys = numeric_function(expr, xs)
        ys[~np.isfinite(ys)] = np.nan
        ax.plot(xs, ys, label="Selected classroom function")
        if tangent:
            d1 = sp.diff(expr, X)
            try:
                y0, m = float(expr.subs(X, point_x)), float(d1.subs(X, point_x))
                tx = np.linspace(point_x - (hi-lo)*.2, point_x + (hi-lo)*.2, 50)
                ax.plot(tx, y0 + m*(tx-point_x), ":", label="Tangent line")
                ax.scatter([point_x], [y0], marker="o", zorder=6, label="Moving point")
            except Exception:
                pass
        if secant:
            try:
                ya, yb = float(expr.subs(X, lo)), float(expr.subs(X, hi))
                m = (yb-ya)/(hi-lo)
                sx = np.linspace(lo, hi, 30)
                ax.plot(sx, ya + m*(sx-lo), "--", label="Secant line")
            except Exception:
                pass
    ax.set_xlabel("Year (historical model) / x (classroom function)")
    ax.set_ylabel(units)
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def dataframe_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def excel_bytes(summary: pd.DataFrame, working: pd.DataFrame, results: pd.DataFrame):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        working.to_excel(writer, sheet_name="Working data", index=False)
        results.to_excel(writer, sheet_name="Calculated years", index=False)
        for sheet_name, data in [("Summary", summary), ("Working data", working), ("Calculated years", results)]:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.set_column(0, max(0, len(data.columns)-1), 22)
    return out.getvalue()


def pdf_bytes(title, metadata, equation, metrics, chart_png, findings, preview, results):
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=landscape(letter), rightMargin=0.45*inch, leftMargin=0.45*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    story.append(Paragraph(metadata, styles["BodyText"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Model:</b> {equation}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Metrics:</b> {metrics}", styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(RLImage(io.BytesIO(chart_png), width=7.5*inch, height=3.5*inch))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Findings and limitations:</b> {findings}", styles["BodyText"]))
    for label, df in [("Data preview", preview), ("Calculated results", results)]:
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"<b>{label}</b>", styles["Heading2"]))
        if df.empty:
            story.append(Paragraph("No rows available.", styles["BodyText"]))
        else:
            data = [list(map(str, df.columns))] + df.head(12).astype(str).values.tolist()
            table = Table(data, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e6e6e6")),
                ("GRID", (0,0), (-1,-1), .35, colors.grey),
                ("FONTSIZE", (0,0), (-1,-1), 7), ("VALIGN", (0,0), (-1,-1), "TOP"),
            ]))
            story.append(table)
    doc.build(story)
    return out.getvalue()


st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")


# -----------------------------
# Descriptive statistics helpers
# -----------------------------
def calculate_descriptive_statistics(values):
    """Return mean, population SD, and sample SD for a sequence of numeric values."""
    import numpy as np
    clean = []
    for value in values:
        try:
            number = float(value)
            if np.isfinite(number):
                clean.append(number)
        except (TypeError, ValueError):
            continue
    if not clean:
        return None
    arr = np.asarray(clean, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "population_sd": float(np.std(arr, ddof=0)),
        "sample_sd": float(np.std(arr, ddof=1)) if arr.size >= 2 else None,
        "minimum": float(np.min(arr)),
        "maximum": float(np.max(arr)),
    }

st.title("📊 " + APP_TITLE)
st.caption("HISPANIC HERITAGE DATA LAB • historical evidence + student-built mathematical models")
st.info("Brazil is included as a Latin American comparison country. The country list is a regional comparison set; it is not a claim that every resident of every country identifies as Hispanic.")
with st.expander("Research question and project choices", expanded=True):
    research_question = st.text_input("Research question", "How has the selected indicator changed over time in the selected country?")
    featured_note = st.text_input("Team's featured countries / indicator", "Choose countries and one indicator below")
    pathway = st.selectbox("Assigned mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    st.caption("You can revise the question and choices as your project develops.")

now_year = date.today().year
start_year, end_year = now_year - 70, now_year - 1
with st.sidebar:
    st.header("Historical data")
    country_label = st.selectbox("Featured country", list(COUNTRIES))
    indicator_label = st.selectbox("Indicator", list(INDICATORS))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    compare_labels = st.multiselect("Compare with countries", [c for c in COUNTRIES if c != country_label], default=[])
    printer_mode = st.checkbox("Printer-friendly report view", value=False)
    st.caption(f"Requested source period: {start_year}–{end_year} (current year minus 70 through minus 1).")

country_code = COUNTRIES[country_label]
indicator_code, units, definition = INDICATORS[indicator_label]
source_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={start_year}:{end_year}&format=json"
st.markdown(f"**Definition:** {definition}  \n**Units:** {units}  \n**Source:** [World Bank API]({source_url})  \n**Retrieved:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}")
if "source_frame" not in st.session_state or st.session_state.get("source_key") != (country_code, indicator_code, start_year, end_year):
    try:
        source = cached_wb_fetch(country_code, indicator_code, start_year, end_year)
        st.session_state.source_frame = source.copy()
        st.session_state.source_key = (country_code, indicator_code, start_year, end_year)
        st.session_state.working_frame = source.copy()
        st.session_state.edited = False
        st.session_state.calculated_years = []
        st.session_state.last_model_key = None
    except RuntimeError as exc:
        st.error(str(exc))
        st.warning("No substitute or random data are used. Try again later or check the API URL and deployment logs.")
        st.stop()
source_frame = st.session_state.source_frame.copy()
source_nonnull = source_frame.dropna(subset=["value"])
st.write(f"**Actual API coverage:** {source_frame['year'].min() if not source_frame.empty else 'none'}–{source_frame['year'].max() if not source_frame.empty else 'none'}; "
         f"**non-missing observations:** {len(source_nonnull)} of {len(source_frame)} returned year rows.")
if source_frame["value"].isna().any():
    st.warning(f"{int(source_frame['value'].isna().sum())} returned years have missing values and are not used in fitting.")
if source_nonnull.empty:
    st.error("No non-missing observations are available for this country and indicator.")
    st.stop()

st.subheader("1. Inspect and edit the working data")
st.write("The source table is kept separately from the editable table. Changes are a scenario, not official World Bank observations.")
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Reset to source"):
        st.session_state.working_frame = source_frame.copy()
        st.session_state.edited = False
        st.session_state.calculated_years = []
        st.rerun()
with c2:
    if st.session_state.edited:
        st.warning("USER-MODIFIED SCENARIO — exported results reflect edits, not only the source.")
    else:
        st.success("Working table currently matches the source table.")
edited_df = st.data_editor(
    st.session_state.working_frame, num_rows="dynamic", use_container_width=True,
    column_config={"year": st.column_config.NumberColumn("Year", format="%d"), "value": st.column_config.NumberColumn(f"Value ({units})", format="%.6g")},
    key=f"editor_{country_code}_{indicator_code}"
)
if not edited_df.equals(st.session_state.working_frame):
    st.session_state.working_frame = edited_df.copy()
    st.session_state.edited = not edited_df.reset_index(drop=True).equals(source_frame.reset_index(drop=True))
    st.session_state.calculated_years = []
working, validation_errors = validate_table(st.session_state.working_frame, indicator_code)
for error in validation_errors:
    st.error(error)
valid_for_fit = not validation_errors and len(working) >= degree + 1 and working["year"].nunique() >= degree + 1
if not valid_for_fit:
    st.warning(f"Fitting is unavailable until the table is valid and has at least {degree+1} distinct observations.")
    st.stop()

try:
    model = poly_fit(working, degree)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
model["horizon"] = horizon
model_key = (country_code, indicator_code, degree, tuple(working["year"]), tuple(np.round(working["value"], 10)))
if st.session_state.get("last_model_key") != model_key:
    st.session_state.calculated_years = []
    st.session_state.last_model_key = model_key
equation = equation_text(model, units)
st.subheader("2. Polynomial fit and model quality")
st.code(equation)
metric_cols = st.columns(5)
metric_cols[0].metric("Degree", degree)
metric_cols[1].metric("Observations", model["count"])
metric_cols[2].metric("In-sample R²", f"{model['r2']:.5f}")
metric_cols[3].metric("In-sample MAE", f"{model['mae']:.5g} {units}")
metric_cols[4].metric("Mean observed value", f"{model['mean']:.5g} {units}")
st.caption(f"Centered base year: {model['base']}. Historical range: {model['min_year']}–{model['max_year']}. Fit metrics are in-sample, not proof of future accuracy.")
residual_df = pd.DataFrame({"year": model["years"], "observed": model["values"], "fitted": model["fitted"], "residual (observed − fitted)": model["residuals"]})
with st.expander("Residuals and chronological holdout"):
    st.dataframe(residual_df, use_container_width=True)
    if model["count"] >= max(12, degree + 5):
        n_test = max(2, int(round(model["count"] * .2)))
        train, test = working.iloc[:-n_test], working.iloc[-n_test:]
        if train["year"].nunique() >= degree + 1:
            hold_model = poly_fit(train, degree)
            hold_pred = poly_predict(hold_model, test["year"].to_numpy())
            st.write(f"**Chronological holdout MAE** on the latest {len(test)} rows: {mean_absolute_error(test['value'], hold_pred):.5g} {units}. Separate from in-sample metrics.")
        else:
            st.write("Not enough training observations for a chronological holdout at this degree.")
    else:
        st.write("Holdout score not shown: more observations are needed for a useful chronological split.")
fig = make_plot(working, model, units, sample_step)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("3. Calculate single-year and batch estimates")
with st.form("single_year_form"):
    single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=min(max(model["max_year"], 1950), 2100), step=1)
    do_single = st.form_submit_button("Calculate Value")
if do_single:
    val = float(poly_predict(model, [single_year])[0])
    st.session_state.calculated_years = [r for r in st.session_state.calculated_years if r.get("kind") != "single"]
    st.session_state.calculated_years.append({"kind":"single", "year":int(single_year), "estimate":val})
if st.session_state.calculated_years:
    for r in st.session_state.calculated_years:
        if r["kind"] == "single":
            cls = "within fitted historical range" if model["min_year"] <= r["year"] <= model["max_year"] else "outside fitted historical range (extrapolation)"
            st.write(f"**{r['year']}: {r['estimate']:.8g} {units}** — {cls}. This is a model estimate, not an observed value.")
with st.form("batch_form"):
    batch_mode = st.radio("Batch input", ["Year range", "Comma-separated custom years"], horizontal=True)
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        range_start = st.number_input("Range start", min_value=1950, max_value=2100, value=max(1950, model["min_year"]), step=1)
    with bcol2:
        range_end = st.number_input("Range end", min_value=1950, max_value=2100, value=min(2100, model["max_year"]), step=1)
    with bcol3:
        range_step = st.slider("Range step", 1, 10, 5)
    custom_years = st.text_input("Custom years", "2000, 2010, 2020")
    do_batch = st.form_submit_button("Calculate batch")
if do_batch:
    try:
        if batch_mode == "Year range":
            if range_end < range_start:
                raise ValueError("Range end must be greater than or equal to range start.")
            years_batch = list(range(int(range_start), int(range_end)+1, int(range_step)))
        else:
            pieces = [p.strip() for p in custom_years.split(",") if p.strip()]
            years_batch = []
            for piece in pieces:
                if not re.fullmatch(r"\d{4}", piece):
                    raise ValueError(f"'{piece}' is not a four-digit integer year.")
                yr = int(piece)
                if not 1950 <= yr <= 2100:
                    raise ValueError(f"Year {yr} is outside the allowed 1950–2100 range.")
                years_batch.append(yr)
            years_batch = list(dict.fromkeys(years_batch))
        if not years_batch:
            raise ValueError("Enter at least one valid year.")
        vals_batch = poly_predict(model, years_batch)
        st.session_state.calculated_years = [r for r in st.session_state.calculated_years if r.get("kind") != "batch"]
        st.session_state.calculated_years.extend([{"kind":"batch","year":int(y),"estimate":float(v)} for y,v in zip(years_batch, vals_batch)])
    except ValueError as exc:
        st.error(f"Batch input error: {exc}")
result_rows = [{"year":r["year"], "estimate":r["estimate"], "units":units, "classification":"within fitted range" if model["min_year"] <= r["year"] <= model["max_year"] else "outside fitted range (extrapolation)", "kind":r["kind"]} for r in st.session_state.calculated_years]
results_df = pd.DataFrame(result_rows, columns=["year","estimate","units","classification","kind"])
if not results_df.empty:
    st.dataframe(results_df, use_container_width=True)
    st.caption("Batch and single-year results use the same centered-time polynomial prediction function.")
    extrap = results_df[results_df["classification"].str.startswith("outside")]
    if not extrap.empty:
        st.warning("Extrapolation can become implausible for high-degree polynomials. Predictions are not clipped; inspect the graph and treat them cautiously.")
st.write(f"**Baseline uncertainty:** in-sample MAE is {model['mae']:.5g} {units}. Residual size is a descriptive error measure, not a validated 95% prediction interval. The model may miss structural changes and should not be used to infer causes.")

if pathway == "Honors Algebra 2":
    st.subheader("4. Algebra 2 interpretation")
    st.write(f"**Domain used for observed fit:** {model['min_year']}–{model['max_year']}; display horizon extends to {model['max_year']+horizon}. The polynomial is mathematically defined for all real t, but its data-supported domain is limited.")
    st.write("**Range:** the fitted output range over the displayed interval can be inspected on the graph; polynomial end behavior is controlled by the leading coefficient and degree. Average rate of change between years a and b is (f(b) − f(a))/(b − a).")
elif pathway == "Precalculus":
    st.subheader("4. Precalculus interpretation")
    st.write("The degree controls possible curvature and turning points; coefficients set scale and shape. A high-degree polynomial can wiggle between observations. Concavity is related to how average rates of change change across intervals.")
else:
    st.subheader("4. Calculus interpretation of the fitted polynomial")
    coeff_expr = sum(sp.Float(float(c), 12) * X**i for i, c in enumerate(model["coeff"]))
    st.write(f"First derivative with respect to centered time t: `{sp.sstr(sp.diff(coeff_expr, X))}`")
    st.write(f"Second derivative with respect to centered time t: `{sp.sstr(sp.diff(coeff_expr, X, 2))}`")
    try:
        critical = [float(z) for z in sp.solve(sp.Eq(sp.diff(coeff_expr, X), 0), X) if z.is_real and 0 <= float(z) <= model['max_year']-model['base']]
        st.write("Numerical critical years: " + (", ".join(f"{model['base']+z:.3f}" for z in critical) if critical else "No real critical points inside the fitted range found."))
    except Exception:
        st.write("Critical point solver could not classify all roots for this degree.")
    st.write("Derivative roots are candidates, not automatically maxima/minima. Extrapolated extrema are not historical evidence.")

st.subheader("5. Compare countries (same indicator)")
comparison_frames = []
for label in compare_labels:
    try:
        cframe = cached_wb_fetch(COUNTRIES[label], indicator_code, start_year, end_year)
        cframe["country"] = label
        comparison_frames.append(cframe.dropna(subset=["value"]))
    except RuntimeError as exc:
        st.warning(f"{label}: {exc}")
if comparison_frames:
    compare_fig, compare_ax = plt.subplots(figsize=(9, 4))
    compare_ax.plot(working["year"], working["value"], marker="o", markersize=3, label=country_label)
    for cf in comparison_frames:
        compare_ax.plot(cf["year"], cf["value"], marker=".", markersize=3, label=cf["country"])
    compare_ax.set_xlabel("Year")
    compare_ax.set_ylabel(units)
    compare_ax.set_title(f"{indicator_label}: coverage may differ by country")
    compare_ax.grid(True, alpha=.25)
    compare_ax.legend(fontsize=8)
    st.pyplot(compare_fig, use_container_width=True)
    plt.close(compare_fig)
    st.caption("Gaps mean the API has no non-missing observation for that country-year. Do not interpret missing values as zero.")
else:
    st.write("Select one or more comparison countries in the sidebar to show same-indicator series.")

st.subheader("6. Function laboratory: context, function, and calculus")
st.write("The historical polynomial is available as one option. You can also explore suggested functions or type your own expression. Custom functions are classroom models and are not automatically fitted to the historical data.")
context = st.selectbox("Context", list(CONTEXTS))
dep_default, dep_unit_default, indep_default, indep_unit_default, suggested, context_note = CONTEXTS[context]
use_historical = st.checkbox("Use generated function from historical curve of best fit", value=False)
function_choice = st.radio("Function source", ["Suggested function", "Build polynomial from parameters", "Enter own function"], horizontal=True, disabled=use_historical)
context_description = st.text_input("Context description", context_note)
dep_name = st.text_input("Dependent variable name", dep_default)
dep_unit = st.text_input("Dependent variable units", dep_unit_default)
indep_name = st.text_input("Independent variable name", indep_default)
indep_unit = st.text_input("Independent variable units", indep_unit_default)
expr_text = suggested
if function_choice == "Build polynomial from parameters" and not use_historical:
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1: a0 = st.number_input("Constant a₀", value=1.0)
    with pcol2: a1 = st.number_input("Linear a₁", value=2.0)
    with pcol3: a2 = st.number_input("Quadratic a₂", value=1.0)
    expr_text = f"{a2}*x**2 + {a1}*x + {a0}"
elif function_choice == "Enter own function" and not use_historical:
    expr_text = st.text_input("Function in x (e.g. x**2 + 3*x, exp(-0.3*x), sin(x))", "x**2 - 4*x + 3")
elif not use_historical:
    st.code(suggested)
if use_historical:
    expr = sum(sp.Float(float(c), 14) * X**i for i,c in enumerate(model["coeff"]))
    # x represents centered years for this option; explain this explicitly.
    st.info(f"Historical-fit function uses x = year − {model['base']}; output units are {units}.")
    dep_unit = units
else:
    try:
        expr = safe_parse(expr_text)
    except ValueError as exc:
        st.error(str(exc))
        expr = None
analysis_cols = st.columns(3)
with analysis_cols[0]: x_min = st.number_input("Analysis interval start", value=0.0, key="analysis_start")
with analysis_cols[1]: x_max = st.number_input("Analysis interval end", value=10.0, key="analysis_end")
with analysis_cols[2]: point_x = st.slider("Current input x", min_value=float(min(x_min, x_max)), max_value=float(max(x_min, x_max)) if x_min != x_max else float(x_min + 1), value=float(min(max(2.0, min(x_min, x_max)), max(x_min, x_max)) if x_min != x_max else x_min + 1), step=0.1)
if x_max <= x_min:
    st.error("Analysis interval end must be greater than interval start.")
    expr = None
if expr is not None:
    try:
        d1, d2, critical_points, inflections, abs_min, abs_max = symbolic_analysis(expr, (x_min, x_max))
        st.markdown(f"**Function:** \(f(x) = {sp.latex(expr)}\)")
        st.markdown(f"**First derivative:** \(f'(x) = {sp.latex(d1)}\)")
        st.markdown(f"**Second derivative:** \(f''(x) = {sp.latex(d2)}\)")
        show_f = st.checkbox("Show function curve", value=True)
        show_d1 = st.checkbox("Show first derivative curve", value=False)
        show_d2 = st.checkbox("Show second derivative curve", value=False)
        calc_col1, calc_col2 = st.columns(2)
        with calc_col1:
            st.markdown("**Instantaneous rate / tangent**")
            try:
                fx = float(expr.subs(X, point_x))
                slope = float(d1.subs(X, point_x))
                st.write(f"{dep_name} at x={point_x:g}: {fx:.8g} {dep_unit}")
                st.write(f"Instantaneous rate: {slope:.8g} {dep_unit}/{indep_unit}")
                st.write(f"Tangent: y = {fx:.6g} + ({slope:.6g})(x − {point_x:g})")
                st.write(f"Interpretation: at this input, the modeled {dep_name.lower()} is changing by about {slope:.5g} {dep_unit} per {indep_unit}.")
            except Exception:
                st.warning("The function or derivative is undefined at the selected point.")
        with calc_col2:
            st.markdown("**Average rate / secant**")
            try:
                fa, fb = float(expr.subs(X, x_min)), float(expr.subs(X, x_max))
                avg = (fb-fa)/(x_max-x_min)
                st.write(f"Average rate from {x_min:g} to {x_max:g}: {avg:.8g} {dep_unit}/{indep_unit}")
                st.write(f"Secant line: y = {fa:.6g} + ({avg:.6g})(x − {x_min:g})")
            except Exception:
                st.warning("The function is undefined at one or both interval endpoints.")
        st.write(f"**Critical-point candidates in interval:** {critical_points if critical_points else 'none classified'}")
        st.write(f"**Inflection points with numerical concavity change check:** {[round(v, 6) for v in inflections] if inflections else 'none verified'}")
        st.write(f"**Absolute minimum on closed interval (candidate/endpoints):** {abs_min}; **absolute maximum:** {abs_max}.")
        st.write("Domain/discontinuities: logarithms require positive arguments; rational functions are undefined where denominators are zero. Curves are plotted with non-finite values removed so lines do not intentionally connect through undefined points.")
        grid_x = np.linspace(x_min, x_max, 600)
        fvals = numeric_function(expr, grid_x)
        d1vals = numeric_function(d1, grid_x)
        d2vals = numeric_function(d2, grid_x)
        fig2, ax2 = plt.subplots(figsize=(9,4.8))
        if show_f:
            fvals[~np.isfinite(fvals)] = np.nan
            ax2.plot(grid_x, fvals, label=f"f(x): {dep_name}")
            try:
                yy, mm = float(expr.subs(X, point_x)), float(d1.subs(X, point_x))
                if math.isfinite(yy) and math.isfinite(mm):
                    ax2.scatter([point_x],[yy],s=55,label="Moving point")
                    ax2.plot(grid_x, yy+mm*(grid_x-point_x), ":", label="Tangent line")
            except Exception: pass
            try:
                fa, fb = float(expr.subs(X,x_min)), float(expr.subs(X,x_max))
                avg = (fb-fa)/(x_max-x_min)
                ax2.plot(grid_x, fa+avg*(grid_x-x_min), "--", label="Secant line")
            except Exception: pass
        if show_d1:
            d1vals[~np.isfinite(d1vals)] = np.nan
            ax2.plot(grid_x, d1vals, label="f'(x)")
        if show_d2:
            d2vals[~np.isfinite(d2vals)] = np.nan
            ax2.plot(grid_x, d2vals, label="f''(x)")
        ax2.set_xlabel(f"{indep_name} ({indep_unit})")
        ax2.set_ylabel(f"Output / derivative values (scale may differ)")
        ax2.grid(True, alpha=.25); ax2.legend(fontsize=8); fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        if pathway == "Calculus":
            st.markdown("**Calculus mode:** critical candidates are classified by checking derivative signs nearby; absolute extrema compare finite endpoint and critical-point values. An inflection candidate is reported only when a numerical test finds a concavity sign change. Numerical tests can miss closely spaced or complicated roots.")
        elif pathway == "Precalculus":
            st.write("**Precalculus mode:** compare slopes and curvature. Turning points occur where the graph changes direction; a zero derivative alone does not prove a turning point.")
        else:
            st.write("**Algebra 2 mode:** inspect domain and range on the selected interval, end behavior, graph shape, and average rate of change.")
        st.write(f"**Context:** {context_description}")
        st.write(f"**Units:** f(x) is in {dep_unit}; f′(x) is {dep_unit} per {indep_unit}; f″(x) is {dep_unit} per {indep_unit} squared.")
    except Exception as exc:
        st.error(f"Could not analyze this function safely: {exc}")

st.subheader("7. Visual simulation")
simulation = st.selectbox("Visual simulation", ["Moving object / projectile", "Walking person with prosthetic leg", "Tank or concentration indicator", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"])
st.write("The simulation is a symbolic classroom illustration, not a physical or medical prediction. The indicator moves with the selected input and the numeric readout uses the selected function.")
if expr is not None:
    try:
        sim_y = float(expr.subs(X, point_x))
        if not math.isfinite(sim_y): raise ValueError("undefined")
        # Scale only for illustration; raw numeric value remains displayed separately.
        scale_value = max(-100, min(100, sim_y))
        progress = max(0.0, min(1.0, (scale_value + 100)/200))
        sim_cols = st.columns([1,2])
        with sim_cols[0]:
            emoji = {"Moving object / projectile":"⚽", "Walking person with prosthetic leg":"🚶", "Tank or concentration indicator":"🧪", "Population / cell colony":"🦠", "Thermometer / weather":"🌡️", "Business / economic indicator":"📈"}[simulation]
            st.markdown(f"<div style='font-size:64px;text-align:center'>{emoji}</div>", unsafe_allow_html=True)
            st.metric("Function output", f"{sim_y:.6g} {dep_unit}")
        with sim_cols[1]:
            st.progress(progress, text=f"Illustrative normalized indicator: {progress:.0%} (display scale clamps at −100 to 100; function value is not clipped)")
            st.write(f"Current input: {point_x:g} {indep_unit}")
            st.write("Change the input slider to synchronize the graph point, tangent, readout, and context explanation.")
    except Exception:
        st.warning("Simulation is paused because the selected function is undefined at this input.")

st.subheader("8. Export the current analysis")
status = "user-modified scenario" if st.session_state.edited else "source-based working data"
summary_df = pd.DataFrame([{
    "research_question": research_question, "featured_country": country_label, "country_code": country_code,
    "indicator": indicator_label, "indicator_code": indicator_code, "definition": definition, "units": units,
    "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"), "requested_start": start_year,
    "requested_end": end_year, "actual_min_year": model["min_year"], "actual_max_year": model["max_year"],
    "observations": model["count"], "degree": degree, "base_year": model["base"], "equation": equation,
    "in_sample_r2": model["r2"], "in_sample_mae": model["mae"], "data_status": status,
    "pathway": pathway, "horizon": horizon, "sample_step": sample_step,
}])
export_working = working.copy()
export_working["data_status"] = status
csv_data = dataframe_csv(results_df if not results_df.empty else export_working)
excel_data = excel_bytes(summary_df, export_working, results_df)
chart_fig = make_plot(working, model, units, sample_step, single_year=(result_rows[-1]["year"] if result_rows else None),
                      single_value=(result_rows[-1]["estimate"] if result_rows else None))
png_buf = io.BytesIO(); chart_fig.savefig(png_buf, format="png", dpi=140, bbox_inches="tight"); plt.close(chart_fig)
findings = (f"Indicator: {definition} In-sample MAE: {model['mae']:.5g} {units}. Polynomial extrapolation can be unstable, "
            "correlation over time does not establish causes, missing years are not zero, and the model does not provide a validated prediction interval.")
pdf_data = pdf_bytes(APP_TITLE, f"Country: {country_label}; indicator: {indicator_label}; requested period: {start_year}–{end_year}; status: {status}; retrieved: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
                     equation, f"R²={model['r2']:.6f} (in-sample), MAE={model['mae']:.6g} (in-sample), n={model['count']}",
                     png_buf.getvalue(), findings, export_working.head(20), results_df)
ex1, ex2, ex3 = st.columns(3)
with ex1:
    st.download_button("Download CSV", csv_data, file_name="hispanic_heritage_results.csv", mime="text/csv")
with ex2:
    st.download_button("Download Excel workbook", excel_data, file_name="hispanic_heritage_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with ex3:
    st.download_button("Download PDF report", pdf_data, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
st.caption("CSV contains calculated-year results when available, otherwise working data. Excel contains Summary, Working data, and Calculated years sheets. PDF contains metadata, model, metrics, chart, findings, data preview, and calculated results.")

if printer_mode:
    st.markdown("---")
    st.subheader("Printer-friendly summary")
    st.write(f"**Question:** {research_question}")
    st.write(f"**Country / indicator:** {country_label} / {indicator_label}")
    st.write(f"**Coverage:** {model['min_year']}–{model['max_year']} ({model['count']} observations)")
    st.write(f"**Equation:** {equation}")
    st.write(f"**In-sample R² / MAE:** {model['r2']:.5f} / {model['mae']:.5g} {units}")
    st.write(f"**Data status:** {status}")
    st.write(findings)

with st.expander("Important source, model, and safety notes"):
    st.markdown("""
- World Bank API data are fetched live and cached for one hour. API failures are shown; the app does not invent replacement data.
- World Bank historical series may have gaps and may be revised. Coverage is reported from returned non-missing observations.
- Polynomial regression is a descriptive curve fit. In-sample metrics describe fit to the same observations used to estimate coefficients.
- High-degree curves can oscillate, and extrapolation outside the observed range is especially uncertain.
- Net migration is a net count, GDP per capita is not household income or average wealth, and crude birth rate is a rate.
- Brazil is included as a Latin American comparison; Hispanic identity and Latin American geography are not interchangeable categories.
- The function lab uses SymPy parsing for familiar mathematical expressions. Do not treat its output as professional medical, engineering, financial, or safety advice.
""")



# -----------------------------
# Mean and standard deviation tool
# -----------------------------
with st.expander("Mean and Standard Deviation", expanded=True):
    st.write(
        "Paste numbers separated by commas, spaces, tabs, or new lines. "
        "You can also choose a numeric column from the app's data table when available."
    )
    pasted_stats_values = st.text_area(
        "Paste values to calculate mean and standard deviation",
        value="12, 15, 18, 20, 25",
        height=100,
        key="descriptive_stats_pasted_values",
        help="Example: 12, 15, 18, 20, 25",
    )
    import re as _stats_re
    _stats_tokens = [t for t in _stats_re.split(r"[,;\s]+", pasted_stats_values.strip()) if t]
    _stats_numbers = []
    _stats_bad_tokens = []
    for _token in _stats_tokens:
        try:
            _number = float(_token)
            if _number == _number and abs(_number) != float("inf"):
                _stats_numbers.append(_number)
            else:
                _stats_bad_tokens.append(_token)
        except ValueError:
            _stats_bad_tokens.append(_token)
    if _stats_bad_tokens:
        st.warning("These entries were skipped because they are not numbers: " + ", ".join(_stats_bad_tokens[:10]))
    _stats_result = calculate_descriptive_statistics(_stats_numbers)
    if _stats_result:
        _stats_col1, _stats_col2, _stats_col3 = st.columns(3)
        _stats_col1.metric("Number of values", _stats_result["count"])
        _stats_col2.metric("Mean", f'{_stats_result["mean"]:.4f}')
        _stats_col3.metric("Population SD", f'{_stats_result["population_sd"]:.4f}')
        if _stats_result["sample_sd"] is not None:
            st.metric("Sample standard deviation", f'{_stats_result["sample_sd"]:.4f}')
        else:
            st.info("Enter at least two numbers to calculate sample standard deviation.")
        st.caption(f'Minimum: {_stats_result["minimum"]:g} · Maximum: {_stats_result["maximum"]:g}')
        st.download_button(
            "Download statistics as CSV",
            data=(
                "Statistic,Value\n"
                f'Count,{_stats_result["count"]}\n'
                f'Mean,{_stats_result["mean"]}\n'
                f'Population standard deviation,{_stats_result["population_sd"]}\n'
                f'Sample standard deviation,{"" if _stats_result["sample_sd"] is None else _stats_result["sample_sd"]}\n'
                f'Minimum,{_stats_result["minimum"]}\n'
                f'Maximum,{_stats_result["maximum"]}\n'
            ),
            file_name="mean_standard_deviation.csv",
            mime="text/csv",
        )
    else:
        st.info("Paste at least one valid number to calculate statistics.")

