from datetime import date

import streamlit as st

import utils as U

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")
U.apply_style()
st.title("📄 Reports")

profile = U.get_profile()
if not profile:
    U.need_profile_message()
    st.stop()

m = U.compute_metrics(profile)
week = U.weekly_plan(m["target"], profile["goal"], profile["diet"], st.session_state.get("plan_seed", 0))

report = f"""AI NUTRITION & MEAL RECOMMENDATION REPORT  ({date.today():%d %b %Y})
{'=' * 60}
Profile : {profile['sex']}, {profile['age']} y, {profile['height_cm']:.0f} cm, {profile['weight_kg']:.1f} kg
Activity: {profile['activity']}   Goal: {profile['goal']}   Diet: {profile['diet']}

BMI            : {m['bmi']:.1f} ({m['bmi_category']})
BMR            : {m['bmr']:.0f} kcal
TDEE           : {m['tdee']:.0f} kcal
Target calories: {m['target']:.0f} kcal/day
Protein / Carbs / Fat: {m['protein_g']:.0f} g / {m['carbs_g']:.0f} g / {m['fat_g']:.0f} g

Disclaimer: educational tool, not medical advice.
"""
st.code(report)

c1, c2 = st.columns(2)
c1.download_button("⬇️ Download summary (.txt)", report, "nutrition_report.txt")
c2.download_button("⬇️ Download 7-day plan (.csv)", week.to_csv(index=False), "meal_plan.csv", "text/csv")

st.subheader("Model comparison (6 regression algorithms)")
with st.spinner("Training models (first run only)..."):
    comp = U.compare_models()
st.dataframe(comp.style.format({"MAE (kcal)": "{:.2f}", "MSE": "{:,.1f}", "RMSE (kcal)": "{:.2f}",
                                "R2": "{:.3f}", "CV R2 (5-fold)": "{:.3f}"}),
             width="stretch", hide_index=True)
