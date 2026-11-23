import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils as U

st.set_page_config(page_title="Visualizations", page_icon="📈", layout="wide")
U.apply_style()
st.title("📈 Visualizations")

profile = U.get_profile()
if not profile:
    U.need_profile_message()
    st.stop()

m = U.compute_metrics(profile)
plan = U.daily_plan(m["target"], profile["goal"], profile["diet"])
week = U.weekly_plan(m["target"], profile["goal"], profile["diet"], st.session_state.get("plan_seed", 0))
bundle = U.get_model()

t1, t2, t3, t4 = st.tabs(["Today's plan", "Weekly calories", "Model", "Dataset"])

with t1:
    a, b = st.columns(2)
    a.plotly_chart(px.bar(plan, x="Meal", y="kcal", title="Calories per meal"), width="stretch")
    macros = [plan["Protein (g)"].sum() * 4, plan["Carbs (g)"].sum() * 4, plan["Fat (g)"].sum() * 9]
    b.plotly_chart(go.Figure(go.Pie(labels=["Protein", "Carbs", "Fat"], values=macros, hole=0.4,
                                    marker_colors=["#E07B53", "#4C8BC4", "#F2C14E"]))
                   .update_layout(title="Macronutrient share of calories"), width="stretch")

with t2:
    daily = week.groupby("Day", sort=False)["kcal"].sum().reset_index()
    fig = px.line(daily, x="Day", y="kcal", markers=True, title="Daily calories across the week")
    fig.add_hline(y=m["target"], line_dash="dash", annotation_text="Target")
    st.plotly_chart(fig, width="stretch")

with t3:
    a, b = st.columns(2)
    sc = px.scatter(x=bundle["y_test"], y=bundle["y_pred"], labels={"x": "Actual (kcal)", "y": "Predicted (kcal)"},
                    title="Predicted vs actual calorie target", opacity=0.5)
    lim = [bundle["y_test"].min(), bundle["y_test"].max()]
    sc.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="red"), name="Ideal"))
    a.plotly_chart(sc, width="stretch")
    imp = (pd.Series(bundle["model"].feature_importances_, index=bundle["columns"])
           .sort_values().tail(8))
    b.plotly_chart(px.bar(imp, orientation="h", title="Top feature importances")
                   .update_layout(showlegend=False), width="stretch")

with t4:
    foods = U.load_foods()
    st.plotly_chart(px.box(foods, x="meal", y="kcal", color="diet", title="Calories by meal and diet type"),
                    width="stretch")
