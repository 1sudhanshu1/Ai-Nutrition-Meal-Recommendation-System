import plotly.graph_objects as go
import streamlit as st

import utils as U

st.set_page_config(page_title="Prediction", page_icon="🤖", layout="wide")
U.apply_style()
st.title("🤖 Calorie Prediction (Machine Learning)")
st.caption("A Random Forest trained on 3,000 synthetic profiles predicts your calorie target; "
           "it is compared with the Mifflin–St Jeor formula.")

profile = U.profile_form("pred")
if not profile:
    st.stop()

with st.spinner("Loading model..."):
    ml_kcal = U.predict_kcal(profile)
formula_kcal = U.calorie_target(profile["sex"], profile["weight_kg"], profile["height_cm"],
                                profile["age"], profile["activity"], profile["goal"])
m = U.compute_metrics(profile)

c = st.columns(4)
c[0].metric("BMI", f"{m['bmi']:.1f}", m["bmi_category"], delta_color="off")
c[1].metric("BMR", f"{m['bmr']:.0f} kcal")
c[2].metric("Formula target", f"{formula_kcal:.0f} kcal")
c[3].metric("ML-predicted target", f"{ml_kcal:.0f} kcal", f"{ml_kcal - formula_kcal:+.0f} vs formula")

fig = go.Figure(go.Bar(x=["Formula", "Random Forest"], y=[formula_kcal, ml_kcal],
                       marker_color=["#4C8BC4", "#2E8B57"]))
fig.update_layout(title="Daily calorie target (kcal)", height=350)
st.plotly_chart(fig, width="stretch")

st.subheader("🍴 Recommended meals for today")
plan = U.daily_plan(ml_kcal, profile["goal"], profile["diet"])
st.dataframe(plan, width="stretch", hide_index=True)
st.write(f"**Plan total:** {plan['kcal'].sum()} kcal")

with st.expander("Model evaluation (test set)"):
    mt = U.get_model()["metrics"]
    st.write(f"Random Forest Regressor (200 trees) · train/test = {mt['n_train']}/{mt['n_test']}")
    e = st.columns(3)
    e[0].metric("R²", f"{mt['r2']:.3f}")
    e[1].metric("MAE", f"{mt['mae']:.1f} kcal")
    e[2].metric("RMSE", f"{mt['rmse']:.1f} kcal")
    st.caption("Labels are synthetic (formula ± 4 % noise): this shows the pipeline works, "
               "not real-world accuracy.")
