"""Shared helpers: health formulas, nutrition dataset, ML model, meal recommender."""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FOODS_CSV = os.path.join(DATA_DIR, "foods.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "calorie_model.pkl")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
ACTIVITY = {
    "Sedentary": 1.2,
    "Lightly active": 1.375,
    "Moderately active": 1.55,
    "Very active": 1.725,
    "Extra active": 1.9,
}
GOAL_ADJ = {"Weight loss": -500, "Maintain": 0, "Weight gain": 400}
MACRO_SPLIT = {  # (protein, carbs, fat) share of calories
    "Weight loss": (0.30, 0.40, 0.30),
    "Maintain": (0.20, 0.50, 0.30),
    "Weight gain": (0.25, 0.50, 0.25),
}
MEAL_SHARE = {"Breakfast": 0.25, "Lunch": 0.35, "Snack": 0.10, "Dinner": 0.30}
ALLOWED = {
    "Vegan": ["Vegan"],
    "Vegetarian": ["Vegan", "Vegetarian"],
    "Non-Veg": ["Vegan", "Vegetarian", "Non-Veg"],
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUTRIENTS = ["kcal", "protein", "carbs", "fat"]

DEFAULT_PROFILE = dict(sex="Male", age=25, height_cm=170, weight_kg=65,
                       activity="Sedentary", goal="Weight loss", diet="Vegetarian")

# name, diet, meal, kcal, protein, carbs, fat, fibre
FOOD_ROWS = [
    ("Vegetable poha", "Vegetarian", "Breakfast", 250, 6, 45, 6, 3),
    ("Moong dal chilla (2)", "Vegetarian", "Breakfast", 220, 14, 28, 6, 5),
    ("Idli with sambar (3)", "Vegan", "Breakfast", 240, 8, 46, 3, 4),
    ("Oats with milk and fruit", "Vegetarian", "Breakfast", 300, 12, 48, 7, 6),
    ("Paneer paratha", "Vegetarian", "Breakfast", 380, 14, 40, 18, 4),
    ("Egg omelette with toast", "Non-Veg", "Breakfast", 330, 20, 24, 17, 2),
    ("Upma", "Vegan", "Breakfast", 230, 6, 38, 7, 3),
    ("Vegetable upma with sambar", "Vegan", "Breakfast", 260, 8, 42, 7, 5),
    ("Smoothie bowl with oats and banana", "Vegetarian", "Breakfast", 320, 10, 55, 7, 6),
    ("Besan chilla with mint chutney", "Vegan", "Breakfast", 240, 12, 30, 7, 5),
    ("Dal, rice and salad", "Vegan", "Lunch", 480, 17, 85, 7, 9),
    ("Rajma chawal", "Vegan", "Lunch", 520, 18, 90, 9, 12),
    ("Paneer bhurji, 2 roti", "Vegetarian", "Lunch", 550, 25, 55, 24, 6),
    ("Chole with 2 roti", "Vegan", "Lunch", 540, 20, 82, 14, 14),
    ("Chicken curry, rice", "Non-Veg", "Lunch", 600, 38, 70, 17, 4),
    ("Grilled fish, rice, veg", "Non-Veg", "Lunch", 520, 36, 58, 14, 5),
    ("Veg pulao with raita", "Vegetarian", "Lunch", 450, 12, 70, 13, 5),
    ("Quinoa salad with chickpeas", "Vegan", "Lunch", 430, 16, 60, 13, 11),
    ("Lentil soup with veg stir-fry", "Vegan", "Lunch", 380, 18, 52, 8, 12),
    ("Curd rice with pickle", "Vegetarian", "Lunch", 400, 11, 62, 11, 3),
    ("Roasted chana (50 g)", "Vegan", "Snack", 180, 10, 28, 3, 8),
    ("Fruit bowl", "Vegan", "Snack", 120, 1, 30, 0, 4),
    ("Greek yogurt with nuts", "Vegetarian", "Snack", 200, 14, 12, 10, 2),
    ("Boiled eggs (2)", "Non-Veg", "Snack", 155, 13, 1, 11, 0),
    ("Peanut chikki", "Vegan", "Snack", 190, 5, 22, 10, 2),
    ("Sprouts salad", "Vegan", "Snack", 140, 9, 22, 2, 6),
    ("Buttermilk with roasted cumin", "Vegetarian", "Snack", 60, 3, 6, 2, 0),
    ("Mixed nuts and seeds (30 g)", "Vegan", "Snack", 190, 6, 7, 16, 3),
    ("Vegetable khichdi", "Vegan", "Dinner", 380, 12, 66, 7, 6),
    ("Palak paneer, 2 roti", "Vegetarian", "Dinner", 480, 22, 50, 20, 8),
    ("Grilled chicken salad", "Non-Veg", "Dinner", 350, 38, 14, 15, 4),
    ("Dal tadka, 2 roti", "Vegan", "Dinner", 420, 16, 68, 9, 11),
    ("Egg curry, 2 roti", "Non-Veg", "Dinner", 470, 24, 52, 18, 6),
    ("Mixed veg curry, 2 roti", "Vegan", "Dinner", 360, 9, 58, 9, 8),
    ("Tofu stir-fry with rice", "Vegan", "Dinner", 430, 20, 56, 13, 5),
    ("Khichdi with cucumber raita", "Vegetarian", "Dinner", 400, 14, 64, 9, 6),
    ("Grilled fish with sauteed veg", "Non-Veg", "Dinner", 380, 34, 18, 18, 5),
]


# --------------------------------------------------------------------------
# Health formulas
# --------------------------------------------------------------------------
def calc_bmi(weight_kg, height_cm):
    return weight_kg / (height_cm / 100) ** 2


def calc_bmr(sex, weight_kg, height_cm, age):
    """Mifflin-St Jeor equation."""
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + 5 if sex == "Male" else base - 161


def calorie_target(sex, weight_kg, height_cm, age, activity, goal):
    return calc_bmr(sex, weight_kg, height_cm, age) * ACTIVITY[activity] + GOAL_ADJ[goal]


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal Weight"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def compute_metrics(p):
    """All dashboard numbers for a profile dict."""
    bmi = calc_bmi(p["weight_kg"], p["height_cm"])
    bmr = calc_bmr(p["sex"], p["weight_kg"], p["height_cm"], p["age"])
    tdee = bmr * ACTIVITY[p["activity"]]
    target = max(1200, tdee + GOAL_ADJ[p["goal"]])
    sex_flag = 1 if p["sex"] == "Male" else 0
    body_fat = max(3.0, 1.2 * bmi + 0.23 * p["age"] - 10.8 * sex_flag - 5.4)  # Deurenberg
    water = p["weight_kg"] * 0.035 + (0.5 if ACTIVITY[p["activity"]] >= 1.55 else 0)
    pr, ca, fa = MACRO_SPLIT[p["goal"]]
    health_score = int(max(0, min(100, 100 - abs(bmi - 22) * 6)))
    return dict(
        bmi=bmi, bmi_category=bmi_category(bmi), bmr=bmr, tdee=tdee, target=target,
        body_fat=body_fat, water=water, health_score=health_score,
        protein_g=pr * target / 4, carbs_g=ca * target / 4, fat_g=fa * target / 9,
        fibre_g=14 * target / 1000,
    )


# --------------------------------------------------------------------------
# Shared UI helpers
# --------------------------------------------------------------------------
def apply_style():
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"]{background:#fff;border-radius:12px;padding:12px 16px;
            box-shadow:0 1px 4px rgba(0,0,0,.08);}
        h1{color:#1F7A4D;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_profile():
    return st.session_state.get("profile")


def profile_form(key):
    """Input form shared by Dashboard and Prediction. Stores the profile in session_state."""
    cur = st.session_state.get("profile", DEFAULT_PROFILE)
    with st.form(f"form_{key}"):
        st.subheader("🧍 Personal Information")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 15, 80, int(cur["age"]))
        height = c2.number_input("Height (cm)", 120.0, 220.0, float(cur["height_cm"]), step=1.0)
        activity = c3.selectbox("Activity level", list(ACTIVITY), list(ACTIVITY).index(cur["activity"]))
        c4, c5, c6 = st.columns(3)
        sex = c4.selectbox("Gender", ["Male", "Female"], ["Male", "Female"].index(cur["sex"]))
        weight = c5.number_input("Weight (kg)", 30.0, 200.0, float(cur["weight_kg"]), step=0.5)
        goal = c6.selectbox("Goal", list(GOAL_ADJ), list(GOAL_ADJ).index(cur["goal"]))
        diet = st.selectbox("Food preference", list(ALLOWED), list(ALLOWED).index(cur["diet"]))
        submitted = st.form_submit_button("Generate my nutrition profile", type="primary")
    if submitted:
        st.session_state["profile"] = dict(sex=sex, age=int(age), height_cm=float(height),
                                           weight_kg=float(weight), activity=activity,
                                           goal=goal, diet=diet)
        st.session_state.pop("plan_seed", None)
    return get_profile()


def need_profile_message():
    st.info("👈 Fill in the form on the **Dashboard** or **Prediction** page first.")


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
@st.cache_data
def load_foods():
    """Load data/foods.csv (created on first run). Replace the CSV to use real data."""
    if not os.path.exists(FOODS_CSV):
        os.makedirs(DATA_DIR, exist_ok=True)
        pd.DataFrame(FOOD_ROWS, columns=["food", "diet", "meal"] + NUTRIENTS + ["fibre"]).to_csv(
            FOODS_CSV, index=False)
    df = pd.read_csv(FOODS_CSV).drop_duplicates().dropna()
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Machine learning
# --------------------------------------------------------------------------
def make_training_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "sex": rng.choice(["Male", "Female"], n),
        "age": rng.integers(18, 61, n),
        "height_cm": rng.integers(145, 196, n),
        "weight_kg": rng.integers(42, 111, n),
        "activity": rng.choice(list(ACTIVITY), n),
        "goal": rng.choice(list(GOAL_ADJ), n),
    })
    df["bmi"] = calc_bmi(df.weight_kg, df.height_cm)
    exact = df.apply(lambda r: calorie_target(r.sex, r.weight_kg, r.height_cm, r.age,
                                              r.activity, r.goal), axis=1)
    df["target_kcal"] = exact * rng.normal(1.0, 0.04, n)  # about +/-4 % noise
    return df


def encode(df, columns=None):
    X = pd.get_dummies(df, columns=["sex", "activity", "goal"])
    return X if columns is None else X.reindex(columns=columns, fill_value=0)


@st.cache_resource
def get_model():
    """Load the saved Random Forest or train + save it on first run."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    data = make_training_data()
    X = encode(data.drop(columns="target_kcal"))
    y = data["target_kcal"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    bundle = dict(model=model, columns=list(X.columns),
                  metrics=dict(r2=r2_score(y_te, pred), mae=mean_absolute_error(y_te, pred),
                               rmse=float(np.sqrt(mean_squared_error(y_te, pred))),
                               n_train=len(X_tr), n_test=len(X_te)),
                  y_test=y_te.values, y_pred=pred)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def predict_kcal(p):
    bundle = get_model()
    row = pd.DataFrame([{**{k: p[k] for k in ["sex", "age", "height_cm", "weight_kg", "activity", "goal"]},
                         "bmi": calc_bmi(p["weight_kg"], p["height_cm"])}])
    return float(bundle["model"].predict(encode(row, bundle["columns"]))[0])


@st.cache_data
def compare_models():
    data = make_training_data()
    X = encode(data.drop(columns="target_kcal"))
    y = data["target_kcal"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Ridge Regression": Ridge(),
        "Linear Regression": LinearRegression(),
        "Random Forest (this project)": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "K-Nearest Neighbours": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    }
    rows = []
    for name, m in models.items():
        pred = m.fit(X_tr, y_tr).predict(X_te)
        mse = mean_squared_error(y_te, pred)
        cv = cross_val_score(m, X, y, cv=5, scoring="r2").mean()
        rows.append({"Model": name, "MAE (kcal)": mean_absolute_error(y_te, pred), "MSE": mse,
                     "RMSE (kcal)": float(np.sqrt(mse)), "R2": r2_score(y_te, pred), "CV R2 (5-fold)": cv})
    return pd.DataFrame(rows).sort_values("RMSE (kcal)").reset_index(drop=True)


# --------------------------------------------------------------------------
# Recommender (content-based, cosine similarity)
# --------------------------------------------------------------------------
def _meal_targets(daily_kcal, goal):
    p, c, f = MACRO_SPLIT[goal]
    out = {}
    for meal, share in MEAL_SHARE.items():
        k = daily_kcal * share
        out[meal] = np.array([[k, p * k / 4, c * k / 4, f * k / 9]])
    return out


def recommend_meal(foods, meal, target, diet, top_k=1, rng=None):
    pool = foods[(foods.meal == meal) & (foods.diet.isin(ALLOWED[diet]))].reset_index(drop=True)
    scaler = MinMaxScaler().fit(np.vstack([pool[NUTRIENTS].values, target]))
    sims = cosine_similarity(scaler.transform(target), scaler.transform(pool[NUTRIENTS].values))[0]
    order = np.argsort(-sims)[:top_k]
    idx = int(rng.choice(order)) if (rng is not None and top_k > 1) else int(order[0])
    best = pool.loc[idx]
    servings = float(min(2.5, max(0.5, round(target[0, 0] / best.kcal * 4) / 4)))
    return {"Meal": meal, "Food": best.food, "Servings": servings,
            "kcal": round(best.kcal * servings), "Protein (g)": round(best.protein * servings),
            "Carbs (g)": round(best.carbs * servings), "Fat (g)": round(best.fat * servings),
            "Similarity": round(float(sims[idx]), 3)}


def daily_plan(daily_kcal, goal, diet):
    foods = load_foods()
    targets = _meal_targets(daily_kcal, goal)
    return pd.DataFrame([recommend_meal(foods, m, t, diet) for m, t in targets.items()])


def weekly_plan(daily_kcal, goal, diet, seed=0):
    foods = load_foods()
    rng = np.random.default_rng(seed)
    targets = _meal_targets(daily_kcal, goal)
    rows = []
    for day in DAYS:
        for meal, t in targets.items():
            rows.append({"Day": day, **recommend_meal(foods, meal, t, diet, top_k=3, rng=rng)})
    return pd.DataFrame(rows)
