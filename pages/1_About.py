import streamlit as st

import utils as U

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
U.apply_style()
st.title("ℹ️ About the Project")

st.markdown(
    """
**AI Nutrition & Meal Recommendation System** — B.Tech CSE, Semester V,
*Foundations of Artificial Intelligence & Machine Learning Laboratory*
(D Y Patil International University, Akurdi, Pune).

### Modules
| Module | Purpose |
|---|---|
| Input | Age, gender, height, weight, activity, goal, diet preference |
| Validation & pre-processing | Range checks, encoding, scaling |
| Feature engineering | BMI, BMR, TDEE, macronutrient targets |
| ML prediction | Random Forest regressor predicts the calorie target |
| Recommendation engine | Content-based filtering with cosine similarity |
| Visualisation | Interactive dashboard and charts |

### Formulas
- BMI = weight(kg) / height(m)²
- BMR (men) = 10·w + 6.25·h − 5·age + 5 ; (women) = … − 161
- TDEE = BMR × activity factor (1.2 / 1.375 / 1.55 / 1.725 / 1.9)
- Goal adjustment: −500 kcal (loss), +400 kcal (gain), 0 (maintain)

### Tech stack
Python · Streamlit · Pandas · NumPy · Scikit-learn · Plotly

> ⚠️ Educational tool, not a medical device. Not suitable for users with clinical
> conditions (e.g. kidney disease, pregnancy).
"""
)
