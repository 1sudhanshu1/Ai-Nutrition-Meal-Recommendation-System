import time

import streamlit as st

import utils as U

st.set_page_config(page_title="Meal Planner", page_icon="🍽️", layout="wide")
U.apply_style()
st.title("🍽️ Weekly Meal Planner")

profile = U.get_profile()
if not profile:
    U.need_profile_message()
    st.stop()

m = U.compute_metrics(profile)
st.caption(f"Target: **{m['target']:.0f} kcal/day** · Goal: **{profile['goal']}** · "
           f"Diet: **{profile['diet']}**")

st.subheader("📅 Today's Recommendation")
today = U.daily_plan(m["target"], profile["goal"], profile["diet"])
cols = st.columns(4)
for col, (_, r) in zip(cols, today.iterrows()):
    col.metric(r["Meal"], f"{r['kcal']} kcal", r["Food"], delta_color="off")
st.dataframe(today, width="stretch", hide_index=True)

st.subheader("🗓️ Your 7-Day Meal Plan")
if st.button("🔄 Regenerate plan"):
    st.session_state["plan_seed"] = int(time.time())
seed = st.session_state.get("plan_seed", 0)
week = U.weekly_plan(m["target"], profile["goal"], profile["diet"], seed)
st.session_state["week_plan"] = week

for day in U.DAYS:
    d = week[week.Day == day]
    with st.expander(f"📌 {day} — {d.kcal.sum()} kcal", expanded=(day == "Monday")):
        for _, r in d.iterrows():
            st.markdown(f"**{r['Meal']}:** {r['Food']} × {r['Servings']} "
                        f"({r['kcal']} kcal · P {r['Protein (g)']} g · C {r['Carbs (g)']} g · F {r['Fat (g)']} g)")
