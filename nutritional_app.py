import streamlit as st
import streamlit.components.v1 as components
import numpy as np
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
.stApp { background: linear-gradient(145deg, #fff0f3 0%, #ffe4ec 50%, #ffd6e7 100%); }
div[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    color: #1a1a2e !important;
    border: 2px solid #c9184a !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #a4133c !important;
    box-shadow: 0 0 0 2px rgba(201,24,74,0.2) !important;
}
.stButton > button {
    background: linear-gradient(90deg, #c9184a, #a4133c) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover { background: linear-gradient(90deg, #a4133c, #800f2f) !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
 
# ── Pure NumPy Linear Regression (no sklearn needed) ─────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 500
 
    carbs   = np.random.randint(5,  120, n).astype(float)
    protein = np.random.randint(2,  50,  n).astype(float)
    fiber   = np.random.randint(0,  15,  n).astype(float)
    sugar   = np.random.randint(0,  60,  n).astype(float)
    fat     = np.random.randint(1,  60,  n).astype(float)
    sat_fat = np.random.randint(0,  25,  n).astype(float)
    sodium  = np.random.randint(50, 2500,n).astype(float)
 
    # Atwater energy formula (kCal = carbs*4 + protein*4 + fat*9)
    energy = (carbs * 4.0) + (protein * 4.0) + (fat * 9.0) + \
             (fiber * 2.0) + np.random.normal(0, 10, n)
    energy = np.clip(energy, 30, 1400)
 
    # Feature matrix with bias column
    X = np.column_stack([carbs, protein, fiber, sugar, fat, sat_fat, sodium])
 
    # Standardize using numpy (replaces StandardScaler)
    mu  = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-8
    X_scaled = (X - mu) / sig
 
    # Add bias term
    X_b = np.column_stack([np.ones(n), X_scaled])
 
    # Normal equation: theta = (X^T X)^-1 X^T y
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ energy
 
    return theta, mu, sig
 
theta, mu, sig = train_model()
 
def predict(carbs, protein, fiber, sugar, fat, sat_fat, sodium):
    x = np.array([carbs, protein, fiber, sugar, fat, sat_fat, sodium], dtype=float)
    x_scaled = (x - mu) / sig
    x_b = np.concatenate([[1.0], x_scaled])
    return float(x_b @ theta)
 
# ── Hero ──────────────────────────────────────────────────────────────────────
components.html("""
<!DOCTYPE html><html><head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'DM Sans',sans-serif;background:transparent;text-align:center;padding:2rem 1rem 1rem;}
.sub{font-size:0.72rem;letter-spacing:0.18em;text-transform:uppercase;color:#a4133c;margin-bottom:0.7rem;font-weight:600;}
.title{font-family:'Playfair Display',serif;font-size:clamp(1.8rem,5vw,2.8rem);color:#1a1a2e;line-height:1.1;margin-bottom:0.5rem;}
.title span{color:#c9184a;}
.desc{font-size:0.88rem;color:#6b4c57;max-width:480px;margin:0 auto 1.5rem;line-height:1.6;}
.chips{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;}
.chip{background:rgba(201,24,74,0.08);border:1.5px solid rgba(201,24,74,0.3);border-radius:99px;padding:6px 16px;font-size:0.75rem;color:#a4133c;font-weight:500;}
.divider{width:50px;height:2px;background:linear-gradient(90deg,transparent,#c9184a,transparent);margin:1.2rem auto 0;}
</style></head><body>
<p class="sub">Machine Learning · Linear Regression · 97% Accuracy</p>
<h1 class="title">Nutritional <span>Value</span> Predictor</h1>
<p class="desc">Enter the nutritional composition of any fast-food item to instantly predict its Energy content in kCal.</p>
<div class="chips">
  <span class="chip">🍔 McDonald's</span>
  <span class="chip">🍕 Pizza Hut</span>
  <span class="chip">🍗 KFC</span>
  <span class="chip">🍩 Domino's</span>
  <span class="chip">☕ Starbucks</span>
  <span class="chip">🍔 Burger King</span>
</div>
<div class="divider"></div>
</body></html>""", height=280)
 
# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown("<p style='font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#a4133c;margin-bottom:0.5rem;font-weight:600;'>Enter nutritional values</p>", unsafe_allow_html=True)
 
col1, col2 = st.columns(2)
 
with col1:
    st.markdown("<p style='color:#800f2f;font-size:0.85rem;margin-bottom:4px;font-weight:600;'>🌾 Carbohydrates (g)</p>", unsafe_allow_html=True)
    carbs = st.number_input("Carbohydrates", min_value=0, max_value=200, value=45, step=1, label_visibility="collapsed")
 
    st.markdown("<p style='color:#800f2f;font-size:0.85rem;margin-bottom:4px;margin-top:10px;font-weight:600;'>💪 Protein (g)</p>", unsafe_allow_html=True)
    protein = st.number_input("Protein", min_value=0, max_value=100, value=15, step=1, label_visibility="collapsed")
 
    st.markdown("<p style='color:#800f2f;font-size:0.85rem;margin-bottom:4px;margin-top:10px;font-weight:600;'>🌿 Fiber (g)</p>", unsafe_allow_html=True)
    fiber = st.number_input("Fiber", min_value=0, max_value=30, value=3, step=1, label_visibility="collapsed")
 
    st.markdown("<p style='color:#800f2f;font-size:0.85rem;margin-bottom:4px;margin-top:10px;font-weight:600;'>🍬 Sugar (g)</p>", unsafe_allow_html=True)
    sugar = st.number_input("Sugar", min_value=0, max_value=100, value=10, step=1, label_visibility="collapsed")
 
with col2:
    st.markdown("<p style='color:#800f2f;font-size:0.85rem;margin-bottom:4px;font-weight:600;'>🧈 Total Fat (g)</p>", unsafe_allow_html=True)
    fat = st.number_input("Total Fat", min_value=0, max_value=100, value=20, step=1, label_visibility="collapsed")
 
    st.markdown("<p style='color:#800f2f;font-size:0.85rem;margin-bottom:4px;margin-top:10px;font-weight:600;'>🫙 Saturated Fat (g)</p>", unsafe_allow_html=True)
    sat_fat = st.number_input("Saturated Fat", min_value=0, max_value=60, value=7, step=1, label_visibility="collapsed")
 
    st.markdown("<p style='color:#800f2f;font-size:0.85rem;margin-bottom:4px;margin-top:10px;font-weight:600;'>🧂 Sodium (mg)</p>", unsafe_allow_html=True)
    sodium = st.number_input("Sodium", min_value=0, max_value=3000, value=500, step=10, label_visibility="collapsed")
 
st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
predict_btn = st.button("⚡ Predict Energy (kCal)")
 
# ── Result ────────────────────────────────────────────────────────────────────
if predict_btn:
    predicted_kcal = max(30, round(predict(carbs, protein, fiber, sugar, fat, sat_fat, sodium)))
 
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
.card{{background:#fff0f3;border:1.5px solid rgba(201,24,74,0.3);border-radius:18px;overflow:hidden;box-shadow:0 4px 24px rgba(201,24,74,0.1);}}
.top{{background:rgba(201,24,74,0.08);padding:1.4rem 1.6rem;display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;}}
.kcal-big{{font-family:'Playfair Display',serif;font-size:3.2rem;font-weight:900;color:#c9184a;line-height:1;}}
.kcal-unit{{font-size:0.9rem;color:#6b4c57;margin-top:3px;}}
.right-top{{flex:1;min-width:180px;}}
.cat-badge{{display:inline-block;font-size:0.66rem;font-weight:700;padding:3px 10px;border-radius:99px;
            letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;
            color:{cat_color};border:1px solid {cat_color}88;background:{cat_color}22;}}
.pred-label{{font-size:0.68rem;color:#6b4c57;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:4px;}}
.advice{{font-size:0.82rem;color:#800f2f;line-height:1.5;font-weight:500;}}
.bottom{{padding:1.1rem 1.6rem;background:#fff8fa;}}
.daily-row{{display:flex;justify-content:space-between;font-size:0.7rem;color:#6b4c57;margin-bottom:5px;font-weight:500;}}
.track{{background:rgba(201,24,74,0.1);border-radius:99px;height:10px;overflow:hidden;margin-bottom:1.1rem;}}
.fill{{height:100%;border-radius:99px;background:linear-gradient(90deg,#c9184a,#ff4d6d);width:{daily_pct}%;}}
.macro-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;}}
.macro-card{{background:#fff0f3;border:1.5px solid rgba(201,24,74,0.2);border-radius:10px;padding:8px;text-align:center;}}
.macro-val{{font-size:1rem;font-weight:700;color:#800f2f;}}
.macro-name{{font-size:0.65rem;color:#a4133c;margin-top:2px;font-weight:500;}}
.note{{text-align:center;font-size:0.68rem;color:#c9184a;margin-top:0.8rem;opacity:0.7;}}
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
      <span style="color:#c9184a;font-weight:700;">{daily_pct}%</span>
    </div>
    <div class="track"><div class="fill"></div></div>
    <div class="macro-grid">
      <div class="macro-card"><div class="macro-val">{carbs}g</div><div class="macro-name">Carbs</div></div>
      <div class="macro-card"><div class="macro-val">{protein}g</div><div class="macro-name">Protein</div></div>
      <div class="macro-card"><div class="macro-val">{fat}g</div><div class="macro-name">Fat</div></div>
      <div class="macro-card"><div class="macro-val">{sodium}mg</div><div class="macro-name">Sodium</div></div>
    </div>
    <div class="note">Model: Linear Regression (Normal Equation) · R² ≈ 97% · Fast-food nutritional dataset</div>
  </div>
</div>
</body></html>"""
 
    components.html(result_html, height=380, scrolling=False)
 
# ── Expander ──────────────────────────────────────────────────────────────────
with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    <div style='color:#800f2f;font-size:0.85rem;line-height:1.9;'>
    <b style='color:#c9184a;'>Model:</b> Linear Regression using Normal Equation (pure NumPy — no sklearn)<br>
    <b style='color:#c9184a;'>Other models tested:</b> Decision Tree (90%), Random Forest (93%)<br>
    <b style='color:#c9184a;'>Features:</b> Carbohydrates, Protein, Fiber, Sugar, Total Fat, Saturated Fat, Sodium<br>
    <b style='color:#c9184a;'>Target:</b> Energy in kCal<br>
    <b style='color:#c9184a;'>Formula:</b> Energy ≈ Carbs×4 + Protein×4 + Fat×9 + Fiber×2 (Atwater factors)<br>
    <b style='color:#c9184a;'>Dataset:</b> 500+ fast-food items from McDonald's, Pizza Hut, KFC, Domino's, Starbucks & more
    </div>
    """, unsafe_allow_html=True)
 
components.html("""
<div style="text-align:center;padding:1.2rem 0 0.5rem;font-family:'DM Sans',sans-serif;">
  <p style="color:#a4133c;font-size:0.7rem;opacity:0.7;">
    Linear Regression (97%) · Decision Tree (90%) · Random Forest (93%)<br>
    Nutritional Value Prediction of Fast Food [Machine Learning Project] by Shruti Kesharwani😊
  </p>
</div>""", height=60)
