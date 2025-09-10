import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
# Load Model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page Config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .main {
            background-color: #f8f9fa;
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
        
        .stNumberInput, .stSelectbox {
            margin-bottom: 15px;
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
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #6a0dad, #8a2be2);
            color: white;
        }
        
        .sidebar-title {
            color: white;
            font-weight: 700;
            font-size: 1.5em;
            margin-bottom: 20px;
        }
        
        .sidebar-info {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title"><span class="heart-icon">❤</span> Heart Disease Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient details below to assess the risk of heart disease</div>', unsafe_allow_html=True)

# Sidebar with enhanced styling
# with st.sidebar:
#     st.markdown('<div class="sidebar-title">ℹ About This App</div>', unsafe_allow_html=True)
#     st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
#     st.info("""
#     This app predicts the likelihood of *Heart Disease* using a Logistic Regression model trained on clinical data.
    
#     - Built with *Streamlit* 🖥
#     - Dataset: Heart Disease Dataset
#     - Developer: *Ashadullah*
#     """)
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     st.markdown("---")
#     st.markdown("### How to use:")
#     st.write("1. Fill in all the patient details")
#     st.write("2. Click the 'Predict' button")
#     st.write("3. Review the results")
    
#     st.markdown("---")
#     st.markdown("### Disclaimer:")
#     st.write("This tool is for informational purposes only and should not replace professional medical advice.")

# Layout with improved spacing
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-label">Age</div>', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=1, max_value=120, value=50, label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Sex</div>', unsafe_allow_html=True)
    sex = st.selectbox("Sex", ["Female", "Male"], label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Chest Pain Type</div>', unsafe_allow_html=True)
    chest_pain = st.selectbox("Chest Pain Type", ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"], label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Resting BP</div>', unsafe_allow_html=True)
    resting_bp = st.number_input("Resting BP", min_value=50, max_value=250, value=120, label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Cholesterol</div>', unsafe_allow_html=True)
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200, label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Fasting Blood Sugar > 120 mg/dl</div>', unsafe_allow_html=True)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], label_visibility="collapsed")

with col2:
    st.markdown('<div class="input-label">Resting ECG</div>', unsafe_allow_html=True)
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Max Heart Rate</div>', unsafe_allow_html=True)
    max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=220, value=150, label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Exercise Angina</div>', unsafe_allow_html=True)
    exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"], label_visibility="collapsed")
    
    st.markdown('<div class="input-label">Oldpeak</div>', unsafe_allow_html=True)
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1, label_visibility="collapsed")
    
    st.markdown('<div class="input-label">ST Slope</div>', unsafe_allow_html=True)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], label_visibility="collapsed")

# Convert inputs to model format
# Encode categorical variables
chest_pain_mapping = {
    "Asymptomatic": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Typical Angina": 3
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

# Predict Button with enhanced styling
if st.button("🔍 Predict Heart Disease Risk"):
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        if prediction == 1:
            st.markdown('<div class="prediction-card error-card"> High Risk: This person may have Heart Disease<br><br>Probability: {:.2f}%</div>'.format(prediction_proba[1]*100), unsafe_allow_html=True)
            st.warning("Recommendation: Please consult with a healthcare professional for further evaluation.")
        else:
            st.markdown('<div class="prediction-card success-card"> Low Risk: This person likely does NOT have Heart Disease<br><br>Probability: {:.2f}%</div>'.format(prediction_proba[0]*100), unsafe_allow_html=True)
            st.success("Recommendation: Maintain a healthy lifestyle with regular check-ups.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("<footer>💜 Built with Streamlit | Ashadhullah</footer>", unsafe_allow_html=True)