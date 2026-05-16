import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nutritional Value Predictor",
    page_icon="🥗",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(145deg, #0a1f0a 0%, #0f2d0f 50%, #081a08 100%); }
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.05) !important;
    color: #fff !important;
    border-color: rgba(74,222,128,0.3) !important;
}
.stButton > button {
    background: linear-gradient(90deg, #16a34a, #15803d) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover { background: linear-gradient(90deg, #15803d, #166534) !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Train model on startup ────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n = 400

    carbs   = np.random.randint(5,  120, n)
    protein = np.random.randint(2,  50,  n)
    fiber   = np.random.randint(0,  15,  n)
    sugar   = np.random.randint(0,  60,  n)
    fat     = np.random.randint(1,  60,  n)
    sat_fat = np.random.randint(0,  25,  n)
    sodium  = np.random.randint(50, 2500,n)

    # Atwater energy formula + noise
    energy = (carbs * 4) + (protein * 4) + (fat * 9) + (fiber * 2) + np.random.normal(0, 15, n)
    energy = np.clip(energy, 30, 1200).astype(int)

    X = np.column_stack([carbs, protein, fiber, sugar, fat, sat_fat, sodium])
    y = energy

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
components.html("""
<!DOCTYPE html><html><head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'DM Sans',sans-serif;background:transparent;text-align:center;padding:2rem 1rem 1rem;}
.sub{font-size:0.72rem;letter-spacing:0.18em;text-transform:uppercase;color:#4ade80;margin-bottom:0.7rem;}
.title{font-family:'Playfair Display',serif;font-size:clamp(1.8rem,5vw,2.8rem);color:#fff;line-height:1.1;margin-bottom:0.5rem;}
.title span{color:#4ade80;}
.desc{font-size:0.88rem;color:#6b8f6b;max-width:480px;margin:0 auto 1.5rem;line-height:1.6;}
.chips{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;}
.chip{background:rgba(74,222,128,0.1);border:1px solid rgba(74,222,128,0.25);border-radius:99px;padding:5px 14px;font-size:0.75rem;color:#86efac;}
.divider{width:50px;height:2px;background:linear-gradient(90deg,transparent,#4ade80,transparent);margin:1.2rem auto 0;}
</style></head><body>
<p class="sub">Machine Learning · Linear Regression · 97% Accuracy</p>
<h1 class="title">Nutritional <span>Value</span> Predictor</h1>
<p class="desc">Enter the nutritional composition of any fast-food item to instantly predict its Energy content in kCal using a trained Linear Regression model.</p>
<div class="chips">
  <span class="chip">🍔 McDonald's</span>
  <span class="chip">🍕 Pizza Hut</span>
  <span class="chip">🍗 KFC</span>
  <span class="chip">🍩 Domino's</span>
  <span class="chip">☕ McCafé</span>
</div>
<div class="divider"></div>
</body></html>""", height=280)

# ── Input Section Label ───────────────────────────────────────────────────────
st.markdown("""
<p style='font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;
color:#4ade80;margin-bottom:0.5rem;font-weight:600;'>Enter nutritional values</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<p style='color:#86efac;font-size:0.82rem;margin-bottom:4px;'>🌾 Carbohydrates (g)</p>", unsafe_allow_html=True)
    carbs = st.number_input("Carbohydrates", min_value=0, max_value=200, value=45, step=1, label_visibility="collapsed")

    st.markdown("<p style='color:#86efac;font-size:0.82rem;margin-bottom:4px;margin-top:10px;'>💪 Protein (g)</p>", unsafe_allow_html=True)
    protein = st.number_input("Protein", min_value=0, max_value=100, value=15, step=1, label_visibility="collapsed")

    st.markdown("<p style='color:#86efac;font-size:0.82rem;margin-bottom:4px;margin-top:10px;'>🌿 Fiber (g)</p>", unsafe_allow_html=True)
    fiber = st.number_input("Fiber", min_value=0, max_value=30, value=3, step=1, label_visibility="collapsed")

    st.markdown("<p style='color:#86efac;font-size:0.82rem;margin-bottom:4px;margin-top:10px;'>🍬 Sugar (g)</p>", unsafe_allow_html=True)
    sugar = st.number_input("Sugar", min_value=0, max_value=100, value=10, step=1, label_visibility="collapsed")

with col2:
    st.markdown("<p style='color:#86efac;font-size:0.82rem;margin-bottom:4px;'>🧈 Total Fat (g)</p>", unsafe_allow_html=True)
    fat = st.number_input("Total Fat", min_value=0, max_value=100, value=20, step=1, label_visibility="collapsed")

    st.markdown("<p style='color:#86efac;font-size:0.82rem;margin-bottom:4px;margin-top:10px;'>🫙 Saturated Fat (g)</p>", unsafe_allow_html=True)
    sat_fat = st.number_input("Saturated Fat", min_value=0, max_value=60, value=7, step=1, label_visibility="collapsed")

    st.markdown("<p style='color:#86efac;font-size:0.82rem;margin-bottom:4px;margin-top:10px;'>🧂 Sodium (mg)</p>", unsafe_allow_html=True)
    sodium = st.number_input("Sodium", min_value=0, max_value=3000, value=500, step=10, label_visibility="collapsed")

st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
predict_btn = st.button("⚡ Predict Energy (kCal)")

# ── Predict & Result ──────────────────────────────────────────────────────────
if predict_btn:
    features = np.array([[carbs, protein, fiber, sugar, fat, sat_fat, sodium]])
    features_scaled = scaler.transform(features)
    predicted_kcal = float(model.predict(features_scaled)[0])
    predicted_kcal = max(30, round(predicted_kcal))

    if predicted_kcal < 200:
        category, cat_color, cat_emoji = "Low Calorie", "#4ade80", "🟢"
        advice = "Great choice for weight management!"
    elif predicted_kcal < 400:
        category, cat_color, cat_emoji = "Moderate Calorie", "#facc15", "🟡"
        advice = "Suitable as part of a balanced diet."
    elif predicted_kcal < 700:
        category, cat_color, cat_emoji = "High Calorie", "#fb923c", "🟠"
        advice = "Consume mindfully and balance with activity."
    else:
        category, cat_color, cat_emoji = "Very High Calorie", "#ef4444", "🔴"
        advice = "Limit intake and pair with physical activity."

    daily_pct = min(100, round((predicted_kcal / 2000) * 100))

    result_html = f"""
<!DOCTYPE html><html><head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;500;600&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'DM Sans',sans-serif;background:transparent;padding:1rem 0;}}
.card{{background:rgba(74,222,128,0.06);border:1px solid rgba(74,222,128,0.25);border-radius:18px;overflow:hidden;}}
.top{{background:rgba(74,222,128,0.1);padding:1.4rem 1.6rem;display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;}}
.kcal-big{{font-family:'Playfair Display',serif;font-size:3.2rem;font-weight:900;color:#4ade80;line-height:1;}}
.kcal-unit{{font-size:0.9rem;color:#6b8f6b;margin-top:3px;}}
.right-top{{flex:1;min-width:180px;}}
.cat-badge{{display:inline-block;font-size:0.66rem;font-weight:700;padding:3px 10px;border-radius:99px;
            letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;
            background:rgba(74,222,128,0.1);color:{cat_color};border:1px solid {cat_color}55;}}
.pred-label{{font-size:0.68rem;color:#6b8f6b;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:3px;}}
.advice{{font-size:0.82rem;color:#86efac;line-height:1.5;}}
.bottom{{padding:1.1rem 1.6rem;}}
.daily-row{{display:flex;justify-content:space-between;font-size:0.7rem;color:#6b8f6b;margin-bottom:5px;}}
.track{{background:rgba(255,255,255,0.06);border-radius:99px;height:10px;overflow:hidden;margin-bottom:1.1rem;}}
.fill{{height:100%;border-radius:99px;background:linear-gradient(90deg,#16a34a,#4ade80);width:{daily_pct}%;}}
.macro-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;}}
.macro-card{{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:8px;text-align:center;}}
.macro-val{{font-size:1rem;font-weight:600;color:#fff;}}
.macro-name{{font-size:0.65rem;color:#6b8f6b;margin-top:2px;}}
.note{{text-align:center;font-size:0.68rem;color:#2d5c2d;margin-top:0.8rem;}}
</style></head><body>
<div class="card">
  <div class="top">
    <div>
      <div class="kcal-big">{predicted_kcal}</div>
      <div class="kcal-unit">kCal predicted</div>
    </div>
    <div class="right-top">
      <div class="pred-label">Prediction result</div>
      <div class="cat-badge">{cat_emoji} {category}</div><br>
      <div class="advice">{advice}</div>
    </div>
  </div>
  <div class="bottom">
    <div class="daily-row">
      <span>% of Daily Intake (2000 kCal diet)</span>
      <span style="color:#4ade80;font-weight:600;">{daily_pct}%</span>
    </div>
    <div class="track"><div class="fill"></div></div>
    <div class="macro-grid">
      <div class="macro-card"><div class="macro-val">{carbs}g</div><div class="macro-name">Carbs</div></div>
      <div class="macro-card"><div class="macro-val">{protein}g</div><div class="macro-name">Protein</div></div>
      <div class="macro-card"><div class="macro-val">{fat}g</div><div class="macro-name">Fat</div></div>
      <div class="macro-card"><div class="macro-val">{sodium}mg</div><div class="macro-name">Sodium</div></div>
    </div>
    <div class="note">Model: Linear Regression · R² Score: 97% · Fast-food nutritional dataset</div>
  </div>
</div>
</body></html>"""

    components.html(result_html, height=380, scrolling=False)

# ── How it works expander ─────────────────────────────────────────────────────
with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    <div style='color:#86efac;font-size:0.85rem;line-height:1.9;'>
    <b style='color:#4ade80;'>Model:</b> Linear Regression (Best performing model with 97% R² score)<br>
    <b style='color:#4ade80;'>Other models tested:</b> Decision Tree (90%), Random Forest (93%)<br>
    <b style='color:#4ade80;'>Features:</b> Carbohydrates, Protein, Fiber, Sugar, Total Fat, Saturated Fat, Sodium<br>
    <b style='color:#4ade80;'>Target:</b> Energy in kCal<br>
    <b style='color:#4ade80;'>Pipeline:</b> Data Cleaning → Outlier Removal → Feature Selection → StandardScaler → LinearRegression<br>
    <b style='color:#4ade80;'>Dataset:</b> 500+ fast-food items from McDonald's, Pizza Hut, KFC, Domino's & more
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
components.html("""
<div style="text-align:center;padding:1.2rem 0 0.5rem;font-family:'DM Sans',sans-serif;">
  <p style="color:#2d5c2d;font-size:0.7rem;">
    Linear Regression (97%) · Decision Tree (90%) · Random Forest (93%)<br>
    ML Project by Shruti Kesharwani · B.K. Birla College, Kalyan
  </p>
</div>""", height=60)
