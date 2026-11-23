import plotly.express as px
import streamlit as st

import utils as U

st.set_page_config(page_title="Dataset Explorer", page_icon="🔎", layout="wide")
U.apply_style()
st.title("🔎 Dataset Explorer")

foods = U.load_foods()
st.caption(f"{len(foods)} foods loaded from `data/foods.csv` — replace that file to use your own dataset.")

c1, c2, c3 = st.columns(3)
diets = c1.multiselect("Diet type", sorted(foods.diet.unique()), default=sorted(foods.diet.unique()))
meals = c2.multiselect("Meal", sorted(foods.meal.unique()), default=sorted(foods.meal.unique()))
kcal_lo, kcal_hi = c3.slider("Calories (kcal)", int(foods.kcal.min()), int(foods.kcal.max()),
                             (int(foods.kcal.min()), int(foods.kcal.max())))
search = st.text_input("Search food name")

df = foods[foods.diet.isin(diets) & foods.meal.isin(meals) & foods.kcal.between(kcal_lo, kcal_hi)]
if search:
    df = df[df.food.str.contains(search, case=False)]

st.dataframe(df, width="stretch", hide_index=True)
st.download_button("⬇️ Download filtered CSV", df.to_csv(index=False), "foods_filtered.csv", "text/csv")

with st.expander("Summary statistics"):
    st.dataframe(df[U.NUTRIENTS + ["fibre"]].describe().round(2), width="stretch")

fig = px.scatter(df, x="carbs", y="protein", size="kcal", color="diet", hover_name="food",
                 title="Protein vs carbohydrates (bubble size = calories)")
st.plotly_chart(fig, width="stretch")
