import streamlit as st

import utils as U

st.set_page_config(page_title="AI Nutrition & Meal Recommendation System",
                   page_icon="🥗", layout="wide")
U.apply_style()

st.title("🥗 AI Nutrition & Meal Recommendation System")
st.caption("Personalised calorie targets and meal plans using Machine Learning")

st.markdown(
    """
Enter a few personal details and the system will:

1. **Calculate** your BMI, BMR (Mifflin–St Jeor) and daily calorie requirement (TDEE)
2. **Predict** your calorie target with a trained **Random Forest** model
3. **Recommend** meals with **content-based filtering** (cosine similarity)
4. **Visualise** calories, macronutrients and a 7-day meal plan
"""
)

c1, c2, c3 = st.columns(3)
c1.metric("ML model", "Random Forest")
c2.metric("Recommender", "Cosine similarity")
c3.metric("Foods in dataset", len(U.load_foods()))

st.success("👈 Open **Dashboard** in the sidebar to get started.")
st.warning("Educational decision-support tool — not a medical device. "
           "It does not replace a dietitian or physician.")
