import plotly.graph_objects as go
import streamlit as st

import utils as U

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
U.apply_style()

st.title("📊 Your Health & Nutrition Dashboard")
st.caption("Fill in your details below to generate a complete, personalised nutrition profile.")

profile = U.profile_form("dash")
if not profile:
    st.stop()

m = U.compute_metrics(profile)

st.subheader("🩺 Your Health Metrics")
c = st.columns(4)
c[0].metric("BMI", f"{m['bmi']:.2f}", m["bmi_category"], delta_color="off")
c[1].metric("Body Fat %", f"{m['body_fat']:.2f} %")
c[2].metric("BMR", f"{m['bmr']:.0f} kcal")
c[3].metric("TDEE", f"{m['tdee']:.0f} kcal")
c = st.columns(4)
c[0].metric("Target Calories", f"{m['target']:.0f} kcal")
c[1].metric("Water Intake", f"{m['water']:.2f} L")
c[2].metric("Goal", profile["goal"])
c[3].metric("Health Score", f"{m['health_score']} / 100")

st.subheader("🥩 Macronutrient Targets")
c = st.columns(4)
c[0].metric("Protein", f"{m['protein_g']:.1f} g")
c[1].metric("Carbohydrates", f"{m['carbs_g']:.1f} g")
c[2].metric("Fat", f"{m['fat_g']:.1f} g")
c[3].metric("Fibre", f"{m['fibre_g']:.1f} g")

left, right = st.columns(2)
fig = go.Figure(go.Pie(labels=["Protein", "Carbs", "Fat"],
                       values=[m["protein_g"] * 4, m["carbs_g"] * 4, m["fat_g"] * 9],
                       hole=0.45, marker_colors=["#E07B53", "#4C8BC4", "#F2C14E"]))
fig.update_layout(title="Calories from each macronutrient", height=350)
left.plotly_chart(fig, width="stretch")

gauge = go.Figure(go.Indicator(mode="gauge+number", value=m["bmi"],
                               title={"text": "BMI"},
                               gauge={"axis": {"range": [10, 40]},
                                      "bar": {"color": "#2E8B57"},
                                      "steps": [{"range": [10, 18.5], "color": "#BBDEFB"},
                                                {"range": [18.5, 25], "color": "#C8E6C9"},
                                                {"range": [25, 30], "color": "#FFE0B2"},
                                                {"range": [30, 40], "color": "#FFCDD2"}]}))
gauge.update_layout(height=350)
right.plotly_chart(gauge, width="stretch")
