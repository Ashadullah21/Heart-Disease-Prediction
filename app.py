import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn, if not available, create a simple prediction function
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not available. Using simple prediction algorithm.")

# Page Config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Create a simple model or prediction function
@st.cache_resource
def create_model():
    if SKLEARN_AVAILABLE:
        # Generate realistic sample data for heart disease prediction
        np.random.seed(42)
        
        # Create more realistic features for heart disease
        n_samples = 2000
        
        # Simulate realistic data ranges
        age = np.random.normal(55, 10, n_samples).clip(20, 80)
        sex = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        chest_pain = np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2])
        resting_bp = np.random.normal(130, 20, n_samples).clip(90, 200)
        cholesterol = np.random.normal(240, 50, n_samples).clip(150, 400)
        fasting_bs = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        resting_ecg = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
        max_hr = np.random.normal(150, 20, n_samples).clip(60, 220)
        exercise_angina = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        oldpeak = np.random.exponential(0.8, n_samples).clip(0, 6)
        st_slope = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
        
        # Create feature matrix
        X = np.column_stack([age, sex, chest_pain, resting_bp, cholesterol, 
                            fasting_bs, resting_ecg, max_hr, exercise_angina, 
                            oldpeak, st_slope])
        
        # Create target variable with some logical relationships
        heart_disease_prob = (
            0.1 * (age - 50) / 30 +
            0.2 * sex +
            0.15 * chest_pain +
            0.1 * (resting_bp - 120) / 40 +
            0.1 * (cholesterol - 200) / 100 +
            0.1 * fasting_bs +
            0.05 * resting_ecg +
            -0.1 * (max_hr - 150) / 30 +
            0.15 * exercise_angina +
            0.2 * oldpeak +
            0.1 * st_slope +
            np.random.normal(0, 0.2, n_samples)
        )
        
        y = (heart_disease_prob > 0.5).astype(int)
        
        # Train logistic regression model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        return model
    else:
        # Simple prediction function without sklearn
        def simple_predict(input_data):
    # Simple heuristic based prediction
            age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope = input_data[0]
    
            risk_score = (
                0.05 * (age - 50) / 30 +          # Reduced from 0.1
                0.1 * sex +                       # Reduced from 0.2
                0.08 * chest_pain +               # Reduced from 0.15
                0.05 * (resting_bp - 120) / 40 +  # Reduced from 0.1
                0.05 * (cholesterol - 200) / 100 + # Reduced from 0.1
                0.05 * fasting_bs +               # Reduced from 0.1
                0.02 * resting_ecg +              # Reduced from 0.05
                -0.05 * (max_hr - 150) / 30 +     # Reduced from -0.1
                0.08 * exercise_angina +          # Reduced from 0.15
                0.1 * oldpeak +                   # Reduced from 0.2
                0.05 * st_slope                   # Reduced from 0.1
            )
    
            probability = 1 / (1 + np.exp(-risk_score))
            prediction = 1 if probability > 0.5 else 0
            
            return prediction, np.array([1 - probability, probability])
        
        return simple_predict

model = create_model()

# Custom CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .main-title {
            text-align: center;
            color: #6a0dad;
            font-size: 2.8em;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        }
        
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 1.2em;
            margin-bottom: 40px;
        }
        
        .prediction-card {
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            font-size: 1.4em;
            font-weight: bold;
            margin: 30px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .success-card {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            color: #155724;
            border: 2px solid #155724;
        }
        
        .error-card {
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            color: #721c24;
            border: 2px solid #721c24;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #6a0dad, #8a2be2);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            font-size: 1.1em;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 20px;
        }
        
        .stButton>button:hover {
            background: linear-gradient(135deg, #8a2be2, #6a0dad);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .demo-warning {
            background-color: #fff3cd;
            color: #856404;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ffeaa7;
            margin: 20px 0;
            text-align: center;
        }
        
        .input-label {
            font-weight: 600;
            color: #444;
            margin-bottom: 5px;
        }
        
        footer {
            text-align: center;
            margin-top: 60px;
            padding: 20px;
            font-size: 0.9em;
            color: #888;
            border-top: 1px solid #eee;
        }
        
        .heart-icon {
            color: #ff4d4d;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title"><span class="heart-icon">❤️</span> Heart Disease Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient details below to assess the risk of heart disease</div>', unsafe_allow_html=True)

# Demo warning
# st.markdown("""
#     <div class="demo-warning">
#         ⚠️ <strong>Demo Mode:</strong> This app uses a simulated model for demonstration purposes. 
#         For actual medical diagnosis, consult healthcare professionals and use clinically validated tools.
#     </div>
# """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('### ℹ️ About This App')
    st.info("""
    This app demonstrates heart disease risk prediction using machine learning.
    
    **Features:**
    - Logistic Regression model
    - 11 clinical parameters
    - Probability scoring
    - Responsive design
    
    **Developer:** Ashadullah
    """)
    
    st.markdown("---")
    st.markdown("### How to use:")
    st.write("1. Fill in patient details")
    st.write("2. Click 'Predict' button")
    st.write("3. Review results")
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer:")
    st.write("This is a DEMO application for educational purposes only. Not for medical use.")

# Input form
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-label">Age</div>', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=20, max_value=100, value=55, label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Sex</div>', unsafe_allow_html=True)
    sex = st.selectbox("Sex", ["Female", "Male"], label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Chest Pain Type</div>', unsafe_allow_html=True)
    chest_pain = st.selectbox("Chest Pain Type", 
                             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], 
                             label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Resting BP (mm Hg)</div>', unsafe_allow_html=True)
    resting_bp = st.number_input("Resting BP", min_value=80, max_value=200, value=130, label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Cholesterol (mg/dl)</div>', unsafe_allow_html=True)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=240, label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Fasting Blood Sugar > 120 mg/dl</div>', unsafe_allow_html=True)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], label_visibility="collapsed")

with col2:
    st.markdown('<div class="input-label">Resting ECG</div>', unsafe_allow_html=True)
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Max Heart Rate</div>', unsafe_allow_html=True)
    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150, label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Exercise Angina</div>', unsafe_allow_html=True)
    exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"], label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Oldpeak</div>', unsafe_allow_html=True)
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.0, value=1.0, step=0.1, label_visibility="collapsed")
    
    st.markdown('<div class="input-label">ST Slope</div>', unsafe_allow_html=True)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], label_visibility="collapsed")

# Convert inputs to model format
chest_pain_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}

resting_ecg_mapping = {
    "Normal": 0,
    "ST": 1,
    "LVH": 2
}

st_slope_mapping = {
    "Up": 0,
    "Flat": 1,
    "Down": 2
}

# Prepare input data
input_data = np.array([
    age,
    1 if sex == "Male" else 0,
    chest_pain_mapping[chest_pain],
    resting_bp,
    cholesterol,
    1 if fasting_bs == "Yes" else 0,
    resting_ecg_mapping[resting_ecg],
    max_hr,
    1 if exercise_angina == "Yes" else 0,
    oldpeak,
    st_slope_mapping[st_slope]
]).reshape(1, -1)

# Predict Button
if st.button("🔍 Predict Heart Disease Risk", type="primary"):
    try:
        if SKLEARN_AVAILABLE:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
        else:
            prediction, prediction_proba = model(input_data)
        
        risk_percentage = prediction_proba[1] * 100
        
        if prediction == 1:
            st.markdown(
                f'<div class="prediction-card error-card">'
                f'🚨 High Risk: Potential Heart Disease Detected<br><br>'
                f'Risk Probability: {risk_percentage:.1f}%</div>',
                unsafe_allow_html=True
            )
            st.warning("""
            **Recommendation:** 
            - Consult a healthcare professional immediately
            - Schedule a comprehensive cardiac evaluation
            - Monitor symptoms closely
            """)
        else:
            st.markdown(
                f'<div class="prediction-card success-card">'
                f'✅ Low Risk: No Significant Heart Disease Detected<br><br>'
                f'Risk Probability: {risk_percentage:.1f}%</div>',
                unsafe_allow_html=True
            )
            st.success("""
            **Recommendation:** 
            - Maintain regular check-ups
            - Continue healthy lifestyle habits
            - Monitor risk factors periodically
            """)
        
        # Show additional info
        with st.expander("📊 Detailed Probability Analysis"):
            st.write(f"**Probability of Heart Disease:** {prediction_proba[1]*100:.1f}%")
            st.write(f"**Probability of No Heart Disease:** {prediction_proba[0]*100:.1f}%")
            st.progress(float(prediction_proba[1]))
            
        st.info("💡 **Note:** This is a demonstration using simulated data. Always consult healthcare professionals for medical advice.")
        
    except Exception as e:
        st.error(f"❌ An error occurred during prediction. Please check your inputs.")

# Footer
st.markdown("---")
st.markdown("<footer>💜 Built with Streamlit | Ashadullah | Educational Demo</footer>", unsafe_allow_html=True)