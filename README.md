# AI Nutrition & Meal Recommendation System

## Run in VS Code
```bash
python -m venv venv
venv\Scripts\activate        # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
The first run creates `data/foods.csv` and trains/saves `models/calorie_model.pkl`.
Deploy: push to GitHub → Streamlit Community Cloud → main file `app.py`.
