import pandas as pd
import numpy as np
import streamlit as st
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import re
import random
import uuid
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(layout="wide", page_title="HealthHub - Complete Health Journey", page_icon="🏥")


# Load data
@st.cache_data
def load_data():
    training = pd.read_csv('Training.csv')
    testing = pd.read_csv('Testing.csv')
    severity_df = pd.read_csv('symptom_severity.csv')
    description_df = pd.read_csv('symptom_Description.csv')
    precaution_df = pd.read_csv('symptom_precaution.csv')

    # Simulated genetic risk factors dataset
    genetic_risk = pd.DataFrame({
        'disease': [d for d in training['prognosis'].unique()],
        'genetic_markers': [
            random.sample(['BRCA1', 'MTHFR', 'ApoE', 'TNFA', 'IL6', 'CYP1A1', 'COMT', 'ACE', 'BDNF', 'FTO'],
                          k=random.randint(1, 5)) for _ in range(len(training['prognosis'].unique()))],
        'risk_score': [random.uniform(0.1, 0.9) for _ in range(len(training['prognosis'].unique()))]
    })

    return training, testing, severity_df, description_df, precaution_df, genetic_risk


training, testing, severity_df, description_df, precaution_df, genetic_risk = load_data()

severity_dict = dict(zip(severity_df.iloc[:, 0], severity_df.iloc[:, 1]))
description_dict = dict(zip(description_df.iloc[:, 0], description_df.iloc[:, 1]))
precaution_dict = dict(zip(precaution_df.iloc[:, 0], precaution_df.iloc[:, 1:].values.tolist()))
genetic_risk_dict = dict(zip(genetic_risk['disease'], zip(genetic_risk['genetic_markers'], genetic_risk['risk_score'])))

cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

# Train and evaluate Decision Tree
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)

# Train and evaluate Random Forest for improved prediction
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(x_train, y_train)
rf_scores = cross_val_score(rf_clf, x_test, y_test, cv=3)

# Train and evaluate SVM
model = SVC(probability=True)
model.fit(x_train, y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

symptoms_dict = {symptom: index for index, symptom in enumerate(x)}

# Initialize session state variables
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'name': '',
        'age': 0,
        'gender': '',
        'height': 0,
        'weight': 0,
        'blood_type': '',
        'allergies': [],
        'medications': [],
        'chronic_conditions': [],
        'family_history': [],
        'lifestyle': {
            'exercise': 'Moderate',
            'diet': 'Balanced',
            'smoking': 'No',
            'alcohol': 'Occasional'
        },
        'dna_analyzed': False,
        'dna_profile': {},
        'wearable_connected': False,
        'wearable_data': {}
    }

if 'health_history' not in st.session_state:
    st.session_state.health_history = []

if 'vital_trends' not in st.session_state:
    # Generate simulated vital trends for demo purposes
    days = 30
    base_date = datetime.now() - timedelta(days=days)
    dates = [(base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days + 1)]

    # Normal ranges with some variation
    heart_rate = [random.randint(60, 100) for _ in range(days + 1)]
    blood_pressure_sys = [random.randint(110, 140) for _ in range(days + 1)]
    blood_pressure_dia = [random.randint(70, 90) for _ in range(days + 1)]
    temperature = [round(random.uniform(36.1, 37.3), 1) for _ in range(days + 1)]
    oxygen = [random.randint(95, 100) for _ in range(days + 1)]
    sleep = [round(random.uniform(5.0, 9.0), 1) for _ in range(days + 1)]
    steps = [random.randint(2000, 15000) for _ in range(days + 1)]

    st.session_state.vital_trends = {
        'dates': dates,
        'heart_rate': heart_rate,
        'blood_pressure_sys': blood_pressure_sys,
        'blood_pressure_dia': blood_pressure_dia,
        'temperature': temperature,
        'oxygen': oxygen,
        'sleep': sleep,
        'steps': steps
    }


# Create sample wearable data for current day
def generate_wearable_data():
    current_time = datetime.now()
    hours = 24
    times = [(current_time - timedelta(hours=hours) + timedelta(hours=i)).strftime("%H:%M") for i in range(hours + 1)]

    # Generate more realistic patterns based on time of day
    heart_rate = []
    for i in range(hours + 1):
        hour = int(times[i].split(":")[0])
        if 0 <= hour < 6:  # sleeping
            heart_rate.append(random.randint(50, 65))
        elif 6 <= hour < 9:  # morning routine
            heart_rate.append(random.randint(65, 85))
        elif 9 <= hour < 12:  # morning work
            heart_rate.append(random.randint(70, 80))
        elif 12 <= hour < 14:  # lunch
            heart_rate.append(random.randint(75, 85))
        elif 14 <= hour < 18:  # afternoon work
            heart_rate.append(random.randint(70, 80))
        elif 18 <= hour < 21:  # evening exercise/activities
            heart_rate.append(random.randint(80, 110))
        else:  # evening relaxation
            heart_rate.append(random.randint(60, 75))

    # Similar patterns for other metrics
    steps_cumulative = [0]
    for i in range(1, hours + 1):
        hour = int(times[i].split(":")[0])
        if 0 <= hour < 6:
            step_increment = random.randint(0, 50)
        elif 6 <= hour < 9:
            step_increment = random.randint(500, 2000)
        elif 9 <= hour < 12:
            step_increment = random.randint(200, 1000)
        elif 12 <= hour < 14:
            step_increment = random.randint(500, 1500)
        elif 14 <= hour < 18:
            step_increment = random.randint(200, 800)
        elif 18 <= hour < 21:
            step_increment = random.randint(1000, 3000)
        else:
            step_increment = random.randint(100, 500)
        steps_cumulative.append(steps_cumulative[-1] + step_increment)

    return {
        'times': times,
        'heart_rate': heart_rate,
        'steps': steps_cumulative,
        'calories': [round(s * 0.05, 1) for s in steps_cumulative],
        'active_minutes': [round(s / 100, 0) for s in steps_cumulative]
    }


# Simulate DNA analysis
def analyze_dna():
    # This is a simulation of DNA analysis results
    blood_biomarkers = {
        'Cholesterol': random.uniform(150, 250),
        'HDL': random.uniform(40, 80),
        'LDL': random.uniform(70, 160),
        'Triglycerides': random.uniform(50, 200),
        'Glucose': random.uniform(70, 120),
        'HbA1c': random.uniform(4.5, 6.5)
    }

    genetic_variants = {
        'BRCA1': random.choice(['Present', 'Absent']),
        'MTHFR': random.choice(['C677T Heterozygous', 'C677T Homozygous', 'Normal']),
        'ApoE': random.choice(['e2/e2', 'e2/e3', 'e3/e3', 'e3/e4', 'e4/e4']),
        'COMT': random.choice(['Val/Val', 'Val/Met', 'Met/Met']),
        'FTO': random.choice(['AA', 'AT', 'TT']),
    }

    # Calculate disease predispositions based on genetic variants
    disease_predispositions = {}
    diseases = training['prognosis'].unique()

    # Simulate genetic risk for common conditions
    for disease in diseases:
        base_risk = random.uniform(0.01, 0.1)  # Base population risk
        genetic_factor = random.uniform(0.8, 4.0)  # Genetic risk multiplier

        # Certain gene variants increase risk for specific conditions
        if genetic_variants['ApoE'] in ['e3/e4', 'e4/e4'] and disease == 'Alzheimers':
            genetic_factor *= 3
        if genetic_variants['BRCA1'] == 'Present' and disease in ['Breast Cancer', 'Ovarian Cancer']:
            genetic_factor *= 5
        if genetic_variants['MTHFR'] != 'Normal' and disease in ['Heart Disease', 'Stroke']:
            genetic_factor *= 1.5

        risk = min(base_risk * genetic_factor, 0.95)  # Cap at 95%
        disease_predispositions[disease] = round(risk, 4)

    return {
        'blood_biomarkers': blood_biomarkers,
        'genetic_variants': genetic_variants,
        'disease_predispositions': disease_predispositions
    }


# Function to calculate condition severity and provide appropriate advice
def calc_condition(exp, days, dna_profile=None):
    sum_severity = sum(severity_dict.get(item, 5) for item in exp)
    severity_score = (sum_severity * days) / (len(exp) + 1)

    recommendation = ""
    if severity_score > 13:
        recommendation = "You should take consultation from a doctor immediately."
        st.error(recommendation)
    elif severity_score > 7:
        recommendation = "You should schedule a doctor's consultation soon."
        st.warning(recommendation)
    else:
        recommendation = "It might not be that bad but you should take precautions."
        st.info(recommendation)

    # Enhanced advice based on DNA profile if available
    if dna_profile and 'disease_predispositions' in dna_profile:
        st.write("### Personalized Risk Assessment")
        for disease, pred_diseases in st.session_state.predicted_diseases.items():
            if disease in dna_profile['disease_predispositions']:
                genetic_risk = dna_profile['disease_predispositions'][disease]
                if genetic_risk > 0.2:
                    st.warning(
                        f"Your genetic profile indicates an elevated risk ({genetic_risk * 100:.1f}%) for {disease}. Consider discussing this with a specialist.")
                elif genetic_risk > 0.1:
                    st.info(
                        f"Your genetic profile shows a moderate risk ({genetic_risk * 100:.1f}%) for {disease}. Monitor your symptoms carefully.")

    return recommendation


# Function to predict disease based on symptoms
def predict_disease(symptoms_exp):
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    # Use both classifiers for more robust prediction
    rf_pred = rf_clf.predict_proba([input_vector])
    dt_pred = clf.predict_proba([input_vector])

    # Combine predictions
    combined_pred = (rf_pred + dt_pred) / 2

    # Get top 3 predictions
    top_indices = combined_pred[0].argsort()[-3:][::-1]
    top_diseases = [(le.inverse_transform([i])[0], combined_pred[0][i]) for i in top_indices]

    return top_diseases


# Function to get disease description from predicted disease
def get_disease_details(disease):
    details = {}
    details['description'] = description_dict.get(disease, "No description available")
    details['precautions'] = precaution_dict.get(disease, ["No specific precautions listed"])

    # Add genetic information if available
    if disease in genetic_risk_dict:
        markers, risk = genetic_risk_dict[disease]
        details['genetic_markers'] = markers
        details['genetic_risk'] = risk
    else:
        details['genetic_markers'] = []
        details['genetic_risk'] = 0

    return details


# Function to generate health recommendations based on symptoms, wearable data and DNA
def generate_recommendations(symptoms, wearable_data, dna_profile, predicted_diseases):
    recommendations = []

    # Basic recommendations based on symptoms
    if symptoms:
        if any(s in ['fatigue', 'weakness', 'tiredness'] for s in symptoms):
            recommendations.append("Consider improving your sleep routine and staying hydrated")

        if any(s in ['headache', 'dizziness', 'vertigo'] for s in symptoms):
            recommendations.append("Monitor your blood pressure and ensure adequate hydration")

    # Recommendations based on wearable data
    if st.session_state.user_profile['wearable_connected'] and wearable_data:
        recent_hr = wearable_data['heart_rate'][-5:]
        avg_hr = sum(recent_hr) / len(recent_hr)

        if avg_hr > 90:
            recommendations.append(
                "Your recent average heart rate is elevated. Consider reducing stress and monitoring your heart health")

        if sum(wearable_data['steps']) < 5000:
            recommendations.append(
                "Your activity level is below recommended levels. Aim for at least 7,500 steps daily")

    # DNA-based recommendations
    if st.session_state.user_profile['dna_analyzed'] and dna_profile:
        if 'blood_biomarkers' in dna_profile:
            if dna_profile['blood_biomarkers']['Cholesterol'] > 200:
                recommendations.append(
                    "Your genetic profile indicates elevated cholesterol levels. Consider a diet lower in saturated fats")

            if dna_profile['blood_biomarkers']['Glucose'] > 100:
                recommendations.append(
                    "Your genetic profile indicates a tendency toward elevated blood sugar. Monitor your carbohydrate intake")

        if 'genetic_variants' in dna_profile:
            if dna_profile['genetic_variants']['MTHFR'] != 'Normal':
                recommendations.append(
                    "Your MTHFR variant may affect folate metabolism. Consider discussing B-vitamin supplementation with your doctor")

    # Add specific recommendations for predicted diseases
    for disease, probability in predicted_diseases:
        if disease in precaution_dict:
            if probability > 0.5:  # Only add specific recommendations for high probability diseases
                precautions = precaution_dict[disease]
                if precautions and any(precautions):
                    recommendations.append(f"For potential {disease.replace('_', ' ')}: {precautions[0]}")

    # Deduplicate and return
    return list(set(recommendations))


# Navigation
def main():
    st.sidebar.image("https://via.placeholder.com/150x100.png?text=HealthHub", width=150)
    st.sidebar.title("HealthHub")
    st.sidebar.subheader("Your Complete Health Journey")

    navigation = st.sidebar.radio(
        "Navigation",
        ["Home", "Health Profile", "Symptom Analysis", "Wearable Integration",
         "DNA Analysis", "Health Dashboard", "Predictive Insights"]
    )

    if navigation == "Home":
        show_home()
    elif navigation == "Health Profile":
        show_health_profile()
    elif navigation == "Symptom Analysis":
        show_symptom_analysis()
    elif navigation == "Wearable Integration":
        show_wearable_integration()
    elif navigation == "DNA Analysis":
        show_dna_analysis()
    elif navigation == "Health Dashboard":
        show_health_dashboard()
    elif navigation == "Predictive Insights":
        show_predictive_insights()


def show_home():
    st.title("Welcome to HealthHub")
    st.subheader("Navigating Your Complete Health Journey")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Your Holistic Health Platform

        HealthHub integrates real-time monitoring, AI-driven analysis, and genetic insights to provide a comprehensive view of your health. Our platform goes beyond isolated parameters to deliver personalized health recommendations.

        **Key Features:**
        * Comprehensive health monitoring
        * Real-time data from wearable devices
        * DNA-based predictive disease analysis
        * Personalized health plans
        * Proactive health management

        *By prognosing disease by diagnose we can save time, and by saving time we can save lives.*
        """)

        if not st.session_state.user_profile['name']:
            st.info("Get started by setting up your Health Profile")
        else:
            st.success(f"Welcome back, {st.session_state.user_profile['name']}!")

            # Show latest health insights
            st.markdown("### Your Latest Health Insights")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

            with metrics_col1:
                if st.session_state.vital_trends:
                    st.metric("Heart Rate", f"{st.session_state.vital_trends['heart_rate'][-1]} BPM")
                else:
                    st.metric("Heart Rate", "-- BPM")

            with metrics_col2:
                if st.session_state.vital_trends:
                    st.metric("Steps Today", f"{st.session_state.vital_trends['steps'][-1]:,}")
                else:
                    st.metric("Steps Today", "--")

            with metrics_col3:
                if st.session_state.vital_trends and 'temperature' in st.session_state.vital_trends:
                    st.metric("Temperature", f"{st.session_state.vital_trends['temperature'][-1]} °C")
                else:
                    st.metric("Temperature", "-- °C")

    with col2:
        st.image("https://via.placeholder.com/300x500.png?text=HealthHub+Overview", width=300)

        # Quick actions
        st.subheader("Quick Actions")
        if st.button("Check My Symptoms"):
            st.switch_page("Symptom Analysis")
        if st.button("View My Health Dashboard"):
            st.switch_page("Health Dashboard")


def show_health_profile():
    st.title("Health Profile")
    st.write("Set up your health profile for personalized insights and recommendations")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Personal Information")
        st.session_state.user_profile['name'] = st.text_input("Full Name",
                                                              st.session_state.user_profile.get('name', ''))
        st.session_state.user_profile['age'] = st.number_input("Age", 0, 120,
                                                               st.session_state.user_profile.get('age', 30))
        gender_options = ["Male", "Female", "Other"]
        current_gender = st.session_state.user_profile.get('gender', 'Male')
        try:
            gender_index = gender_options.index(current_gender) if current_gender in gender_options else 0
        except ValueError:
            gender_index = 0  # Default to first option if value not found

        st.session_state.user_profile['gender'] = st.selectbox("Gender", gender_options, index=gender_index)

        st.subheader("Body Metrics")
        st.session_state.user_profile['height'] = st.number_input("Height (cm)", 0, 250,
                                                                  st.session_state.user_profile.get('height', 170))
        st.session_state.user_profile['weight'] = st.number_input("Weight (kg)", 0, 300,
                                                                  st.session_state.user_profile.get('weight', 70))

        if st.session_state.user_profile['height'] > 0 and st.session_state.user_profile['weight'] > 0:
            bmi = st.session_state.user_profile['weight'] / ((st.session_state.user_profile['height'] / 100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")

            if bmi < 18.5:
                st.info("You are underweight")
            elif bmi < 25:
                st.success("You are in a healthy weight range")
            elif bmi < 30:
                st.warning("You are overweight")
            else:
                st.error("You are in the obese range")

    with col2:
        st.subheader("Medical Information")
        st.session_state.user_profile['blood_type'] = st.selectbox("Blood Type",
                                                                   ["Unknown", "A+", "A-", "B+", "B-", "AB+", "AB-",
                                                                    "O+", "O-"],
                                                                   index=["Unknown", "A+", "A-", "B+", "B-", "AB+",
                                                                          "AB-", "O+", "O-"].index(
                                                                       st.session_state.user_profile.get('blood_type',
                                                                                                         'Unknown')) if st.session_state.user_profile.get(
                                                                       'blood_type') else 0
                                                                   )

        allergies = st.text_area("Allergies (one per line)",
                                 "\n".join(st.session_state.user_profile.get('allergies', [])))
        st.session_state.user_profile['allergies'] = [a.strip() for a in allergies.split("\n") if a.strip()]

        medications = st.text_area("Current Medications (one per line)",
                                   "\n".join(st.session_state.user_profile.get('medications', [])))
        st.session_state.user_profile['medications'] = [m.strip() for m in medications.split("\n") if m.strip()]

        chronic = st.text_area("Chronic Conditions (one per line)",
                               "\n".join(st.session_state.user_profile.get('chronic_conditions', [])))
        st.session_state.user_profile['chronic_conditions'] = [c.strip() for c in chronic.split("\n") if c.strip()]

        st.subheader("Lifestyle")
        st.session_state.user_profile['lifestyle']['exercise'] = st.selectbox("Exercise Level",
                                                                              ["Sedentary", "Light", "Moderate",
                                                                               "Active", "Very Active"],
                                                                              index=["Sedentary", "Light", "Moderate",
                                                                                     "Active", "Very Active"].index(
                                                                                  st.session_state.user_profile.get(
                                                                                      'lifestyle', {}).get('exercise',
                                                                                                           'Moderate')) if st.session_state.user_profile.get(
                                                                                  'lifestyle', {}).get(
                                                                                  'exercise') else 2
                                                                              )

        st.session_state.user_profile['lifestyle']['diet'] = st.selectbox("Diet Type",
                                                                          ["Standard", "Vegetarian", "Vegan", "Keto",
                                                                           "Paleo", "Mediterranean", "Other"],
                                                                          index=["Standard", "Vegetarian", "Vegan",
                                                                                 "Keto", "Paleo", "Mediterranean",
                                                                                 "Other"].index(
                                                                              st.session_state.user_profile.get(
                                                                                  'lifestyle', {}).get('diet',
                                                                                                       'Standard')) if st.session_state.user_profile.get(
                                                                              'lifestyle', {}).get('diet') else 0
                                                                          )

        st.session_state.user_profile['lifestyle']['smoking'] = st.selectbox("Smoking",
                                                                             ["No", "Occasionally", "Regularly",
                                                                              "Former smoker"],
                                                                             index=["No", "Occasionally", "Regularly",
                                                                                    "Former smoker"].index(
                                                                                 st.session_state.user_profile.get(
                                                                                     'lifestyle', {}).get('smoking',
                                                                                                          'No')) if st.session_state.user_profile.get(
                                                                                 'lifestyle', {}).get('smoking') else 0
                                                                             )

        st.session_state.user_profile['lifestyle']['alcohol'] = st.selectbox("Alcohol Consumption",
                                                                             ["None", "Occasional", "Moderate",
                                                                              "Regular"],
                                                                             index=["None", "Occasional", "Moderate",
                                                                                    "Regular"].index(
                                                                                 st.session_state.user_profile.get(
                                                                                     'lifestyle', {}).get('alcohol',
                                                                                                          'Occasional')) if st.session_state.user_profile.get(
                                                                                 'lifestyle', {}).get('alcohol') else 1
                                                                             )

    if st.button("Save Profile"):
        if not st.session_state.user_profile['name']:
            st.error("Please enter your name")
        else:
            st.success("Profile saved successfully!")
            # Generate user ID if not exists
            if 'user_id' not in st.session_state.user_profile:
                st.session_state.user_profile['user_id'] = str(uuid.uuid4())


def show_symptom_analysis():
    st.title("Symptom Analysis & Disease Prediction")

    # Check if user profile exists
    if not st.session_state.user_profile['name']:
        st.warning("Please set up your health profile first for more accurate analysis")

    # Initialize session state for symptom analysis if not present
    if 'symptom_step' not in st.session_state:
        st.session_state.symptom_step = 'initial'
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = []
    if 'predicted_diseases' not in st.session_state:
        st.session_state.predicted_diseases = {}
    if 'current_recommendations' not in st.session_state:
        st.session_state.current_recommendations = []

    if st.session_state.symptom_step == 'initial':
        st.write("Please select the symptoms you are experiencing:")

        # Create a more user-friendly symptom selection interface
        symptom_options = sorted([s.replace('_', ' ').title() for s in symptoms_dict.keys()])

        # Search bar for symptoms
        symptom_search = st.text_input("Search for symptoms")
        filtered_symptoms = [s for s in symptom_options if symptom_search.lower() in s.lower()]

        # Organize symptoms by body system for easier selection
        body_systems = {
            "General": ["Fatigue", "Weakness", "Fever", "Weight Loss", "Weight Gain", "Chills", "Sweating", "Malaise"],
            "Head & Neurological": ["Headache", "Dizziness", "Confusion", "Memory Loss", "Seizures", "Fainting"],
            "Eyes": ["Blurred Vision", "Eye Pain", "Eye Redness", "Sensitivity To Light"],
            "Ear, Nose & Throat": ["Sore Throat", "Runny Nose", "Congestion", "Ear Pain", "Hearing Loss", "Sneezing"],
            "Respiratory": ["Cough", "Shortness Of Breath", "Wheezing", "Chest Pain", "Fast Breathing"],
            "Cardiovascular": ["Chest Pain", "Palpitations", "Swelling", "High Blood Pressure"],
            "Gastrointestinal": ["Nausea", "Vomiting", "Diarrhea", "Constipation", "Abdominal Pain", "Bloating"],
            "Musculoskeletal": ["Joint Pain", "Muscle Pain", "Back Pain", "Stiffness", "Swelling"],
            "Skin": ["Rash", "Itching", "Bruising", "Discoloration", "Dryness"],
            "Psychological": ["Anxiety", "Depression", "Sleep Disturbance", "Irritability", "Mood Swings"]
        }

        selected_category = st.selectbox("Filter by body system", ["All Symptoms"] + list(body_systems.keys()))

        if selected_category == "All Symptoms":
            display_symptoms = filtered_symptoms
        else:
            display_symptoms = [s for s in filtered_symptoms if
                                s in [sym.title() for sym in body_systems[selected_category]]]

        # Multi-select for symptoms
        selected = st.multiselect("Select symptoms", display_symptoms, default=st.session_state.selected_symptoms)
        st.session_state.selected_symptoms = selected

        # Duration of symptoms
        duration = st.slider("For how many days have you been experiencing these symptoms?", 1, 90, 7)
        st.session_state.symptom_duration = duration

        # Severity of symptoms
        severity_options = {
            "Mild": "Symptoms are present but don't significantly affect daily activities",
            "Moderate": "Symptoms interfere with some daily activities",
            "Severe": "Symptoms significantly limit ability to perform daily activities"
        }
        selected_severity = st.radio("Severity of symptoms",
                                     list(severity_options.keys()),
                                     format_func=lambda x: f"{x}: {severity_options[x]}"
                                     )
        st.session_state.symptom_severity = selected_severity

        # Proceed to analysis
        if st.button("Analyze Symptoms") and st.session_state.selected_symptoms:
            # Convert selected symptoms back to the format used in the dataset
            formatted_symptoms = [s.lower().replace(' ', '_') for s in st.session_state.selected_symptoms]

            # Get disease predictions
            st.session_state.predicted_diseases = predict_disease(formatted_symptoms)

            # Generate recommendations
            wearable_data = st.session_state.vital_trends if st.session_state.user_profile[
                'wearable_connected'] else None
            dna_profile = st.session_state.user_profile['dna_profile'] if st.session_state.user_profile[
                'dna_analyzed'] else None

            st.session_state.current_recommendations = generate_recommendations(
                formatted_symptoms,
                wearable_data,
                dna_profile,
                st.session_state.predicted_diseases
            )

            # Log this health event
            health_event = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'symptoms': st.session_state.selected_symptoms,
                'duration': st.session_state.symptom_duration,
                'severity': st.session_state.symptom_severity,
                'predictions': [(disease, f"{prob:.1%}") for disease, prob in st.session_state.predicted_diseases],
                'recommendations': st.session_state.current_recommendations
            }

            st.session_state.health_history.append(health_event)
            st.session_state.symptom_step = 'results'
            st.rerun()

    elif st.session_state.symptom_step == 'results':
        # Display the analysis results
        st.subheader("Analysis Results")

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.session_state.predicted_diseases:
                st.write("### Potential Conditions")

                for disease, probability in st.session_state.predicted_diseases:
                    formatted_disease = disease.replace('_', ' ').title()

                    # Create expandable section for each disease
                    with st.expander(f"{formatted_disease} ({probability:.1%} probability)"):
                        details = get_disease_details(disease)

                        st.write("**Description:**")
                        st.write(details['description'])

                        st.write("**Precautions:**")
                        for i, precaution in enumerate(details['precautions']):
                            if precaution and precaution.strip():
                                st.write(f"- {precaution}")

                        # Show genetic information if available
                        if st.session_state.user_profile['dna_analyzed'] and 'genetic_markers' in details and details[
                            'genetic_markers']:
                            st.write("**Genetic Factors:**")
                            if isinstance(details['genetic_markers'], list):
                                for marker in details['genetic_markers']:
                                    st.write(f"- {marker}")
                            else:
                                st.write(f"- {details['genetic_markers']}")

                # Calculate condition severity
                st.write("### Recommendation")
                formatted_symptoms = [s.lower().replace(' ', '_') for s in st.session_state.selected_symptoms]
                calc_condition(
                    formatted_symptoms,
                    st.session_state.symptom_duration,
                    st.session_state.user_profile['dna_profile'] if st.session_state.user_profile[
                        'dna_analyzed'] else None
                )

            # Show personalized recommendations
            if st.session_state.current_recommendations:
                st.write("### Personalized Health Recommendations")
                for i, rec in enumerate(st.session_state.current_recommendations):
                    st.write(f"- {rec}")

            # Options for saving or new analysis
            if st.button("Start New Analysis"):
                st.session_state.symptom_step = 'initial'
                st.session_state.selected_symptoms = []
                st.rerun()

        with col2:
            # Visual representation of diagnosis confidence
            if st.session_state.predicted_diseases:
                st.write("### Diagnosis Confidence")

                fig = go.Figure()

                diseases = [d[0].replace('_', ' ').title() for d in st.session_state.predicted_diseases]
                probabilities = [d[1] for d in st.session_state.predicted_diseases]

                fig.add_trace(go.Bar(
                    x=probabilities,
                    y=diseases,
                    orientation='h',
                    marker=dict(
                        color=['rgba(55, 83, 109, 0.7)', 'rgba(26, 118, 255, 0.7)', 'rgba(56, 192, 255, 0.7)'],
                        line=dict(
                            color=['rgba(55, 83, 109, 1.0)', 'rgba(26, 118, 255, 1.0)', 'rgba(56, 192, 255, 1.0)'],
                            width=2)
                    )
                ))

                fig.update_layout(
                    title='Diagnosis Probability',
                    xaxis=dict(
                        title='Probability',
                        tickformat='.0%',
                        range=[0, 1]
                    ),
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

            # Additional information or contextual data
            st.write("### Health Context")

            # Show relevant vital signs if wearable is connected
            if st.session_state.user_profile['wearable_connected']:
                st.write("**Recent Vital Signs:**")

                # Heart rate trend
                hr_data = st.session_state.vital_trends['heart_rate'][-7:]
                avg_hr = sum(hr_data) / len(hr_data)

                st.metric("Average Heart Rate (Week)", f"{avg_hr:.0f} BPM")

                # BP if available
                if 'blood_pressure_sys' in st.session_state.vital_trends:
                    sys_data = st.session_state.vital_trends['blood_pressure_sys'][-1]
                    dia_data = st.session_state.vital_trends['blood_pressure_dia'][-1]
                    st.metric("Blood Pressure", f"{sys_data}/{dia_data} mmHg")
            else:
                st.info("Connect a wearable device for enhanced health analysis")

            # Show DNA insights if available
            if st.session_state.user_profile['dna_analyzed']:
                st.write("**Genetic Insights Applied**")
                st.success("Your analysis has been enhanced with your DNA profile")
            else:
                st.info("Add DNA analysis for personalized genetic insights")


def show_wearable_integration():
    st.title("Wearable Device Integration")
    st.write("Connect and monitor your health data from wearable devices")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Device Management")

        # Simulated device connection
        wearable_types = ["Fitness Tracker", "Smartwatch", "Heart Monitor", "Sleep Tracker", "Blood Pressure Monitor"]
        selected_device = st.selectbox("Select Device Type", wearable_types)

        device_brands = {
            "Fitness Tracker": ["Fitbit", "Garmin", "Xiaomi", "Samsung"],
            "Smartwatch": ["Apple Watch", "Samsung Galaxy Watch", "Fitbit Versa", "Garmin Venu"],
            "Heart Monitor": ["Polar", "Garmin", "Whoop", "AliveCor"],
            "Sleep Tracker": ["Oura Ring", "Withings", "Beddit", "Emfit"],
            "Blood Pressure Monitor": ["Omron", "Withings", "iHealth", "Qardio"]
        }

        selected_brand = st.selectbox("Select Brand", device_brands[selected_device])

        # Connection status
        if st.button("Connect Device"):
            st.session_state.user_profile['wearable_connected'] = True
            st.session_state.user_profile['wearable_info'] = {
                'type': selected_device,
                'brand': selected_brand,
                'connected_date': datetime.now().strftime("%Y-%m-%d")
            }

            # Generate simulated wearable data
            if 'wearable_data' not in st.session_state.user_profile or not st.session_state.user_profile[
                'wearable_data']:
                st.session_state.user_profile['wearable_data'] = generate_wearable_data()

            st.success(f"Successfully connected to {selected_brand} {selected_device}")

        if st.session_state.user_profile['wearable_connected']:
            st.info(
                f"Currently connected to: {st.session_state.user_profile['wearable_info']['brand']} {st.session_state.user_profile['wearable_info']['type']}")

            if st.button("Disconnect Device"):
                st.session_state.user_profile['wearable_connected'] = False
                st.success("Device disconnected")
                st.rerun()

        # Data sync options
        if st.session_state.user_profile['wearable_connected']:
            st.subheader("Data Synchronization")
            sync_frequency = st.radio("Sync Frequency", ["Real-time", "Hourly", "Daily"])

            metrics_to_track = st.multiselect("Metrics to Track",
                                              ["Heart Rate", "Steps", "Sleep", "Blood Pressure", "Temperature",
                                               "Oxygen Saturation", "Calories", "Activity"],
                                              default=["Heart Rate", "Steps", "Sleep"]
                                              )

            if st.button("Update Settings"):
                st.session_state.user_profile['wearable_settings'] = {
                    'sync_frequency': sync_frequency,
                    'metrics': metrics_to_track
                }
                st.success("Wearable settings updated")

    with col2:
        if st.session_state.user_profile['wearable_connected']:
            st.subheader("Today's Health Metrics")

            # Display simulated current day data
            wearable_data = st.session_state.user_profile.get('wearable_data', {})

            if wearable_data:
                # Create metrics for last recorded values
                metrics_col1, metrics_col2 = st.columns(2)

                with metrics_col1:
                    if 'heart_rate' in wearable_data:
                        st.metric("Current Heart Rate", f"{wearable_data['heart_rate'][-1]} BPM")

                    if 'steps' in wearable_data:
                        st.metric("Steps Today", f"{wearable_data['steps'][-1]:,}")

                with metrics_col2:
                    if 'calories' in wearable_data:
                        st.metric("Calories Burned", f"{wearable_data['calories'][-1]} kcal")

                    if 'active_minutes' in wearable_data:
                        st.metric("Active Minutes", f"{wearable_data['active_minutes'][-1]} min")

                # Create heart rate chart
                if 'heart_rate' in wearable_data and 'times' in wearable_data:
                    st.subheader("Heart Rate Today")
                    hr_fig = px.line(
                        x=wearable_data['times'],
                        y=wearable_data['heart_rate'],
                        labels={'x': 'Time', 'y': 'Heart Rate (BPM)'}
                    )
                    hr_fig.update_layout(height=250)
                    st.plotly_chart(hr_fig, use_container_width=True)

                # Create steps chart
                if 'steps' in wearable_data and 'times' in wearable_data:
                    st.subheader("Steps Today")
                    steps_fig = px.line(
                        x=wearable_data['times'],
                        y=wearable_data['steps'],
                        labels={'x': 'Time', 'y': 'Cumulative Steps'}
                    )
                    steps_fig.update_layout(height=250)
                    st.plotly_chart(steps_fig, use_container_width=True)
            else:
                st.info("Connecting to device. Please wait for data to sync.")

            # Health insights based on wearable data
            st.subheader("Health Insights")
            if wearable_data and 'heart_rate' in wearable_data:
                hr_data = wearable_data['heart_rate']
                avg_hr = sum(hr_data) / len(hr_data)

                if avg_hr > 90:
                    st.warning(
                        "Your average heart rate today is elevated. Consider relaxation techniques or check with a healthcare provider if this persists.")
                elif avg_hr < 60:
                    st.info(
                        "Your average heart rate today is on the lower side. This could be normal for physically fit individuals.")
                else:
                    st.success("Your heart rate is within normal range today.")

                if 'steps' in wearable_data:
                    steps = wearable_data['steps'][-1]
                    if steps < 4000:
                        st.warning(
                            "You're below the recommended daily step count. Try to aim for at least 7,500 steps for better health.")
                    elif steps > 10000:
                        st.success("Great job! You've exceeded 10,000 steps today.")
        else:
            st.info("Connect a wearable device to see your health metrics")
            st.image("https://via.placeholder.com/400x300.png?text=Wearable+Integration", width=400)


def show_dna_analysis():
    st.title("DNA Analysis & Genetic Insights")
    st.write("Leverage genetic testing for personalized health insights and disease risk assessment")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("DNA Analysis Status")

        if not st.session_state.user_profile['dna_analyzed']:
            st.info("You haven't uploaded any DNA analysis yet")

            st.subheader("Upload or Request DNA Analysis")
            analysis_option = st.radio(
                "How would you like to proceed?",
                ["Upload existing DNA test results", "Request a DNA testing kit", "Simulate DNA analysis for demo"]
            )

            if analysis_option == "Upload existing DNA test results":
                uploaded_file = st.file_uploader("Upload your DNA test results (accepted formats: .txt, .csv, .vcf)",
                                                 type=['txt', 'csv', 'vcf'])

                if uploaded_file is not None:
                    st.success("File uploaded successfully. Processing your DNA data...")
                    # In a real app, we'd process the file here

                    # For the demo, we'll simulate DNA analysis
                    if st.button("Process DNA Data"):
                        st.session_state.user_profile['dna_analyzed'] = True
                        st.session_state.user_profile['dna_profile'] = analyze_dna()
                        st.session_state.user_profile['dna_source'] = "Uploaded File"
                        st.success("DNA analysis completed!")
                        st.rerun()

            elif analysis_option == "Request a DNA testing kit":
                st.write("Complete the form below to request a DNA testing kit by mail:")

                address = st.text_area("Shipping Address")
                kit_type = st.selectbox("Kit Type", ["Health + Ancestry", "Health Only", "Ancestry Only"])

                if st.button("Request Kit") and address:
                    st.success("Your DNA testing kit has been ordered and will arrive in 3-5 business days")

            elif analysis_option == "Simulate DNA analysis for demo":
                if st.button("Simulate DNA Analysis"):
                    st.session_state.user_profile['dna_analyzed'] = True
                    st.session_state.user_profile['dna_profile'] = analyze_dna()
                    st.session_state.user_profile['dna_source'] = "Simulation"
                    st.success("DNA analysis simulation completed!")
                    st.rerun()

        else:
            st.success("DNA analysis completed")
            st.info(f"Source: {st.session_state.user_profile['dna_source']}")

            if st.button("Reset DNA Analysis"):
                st.session_state.user_profile['dna_analyzed'] = False
                st.session_state.user_profile['dna_profile'] = {}
                st.session_state.user_profile['dna_source'] = ""
                st.success("DNA data has been reset")
                st.rerun()

            # Display biomarkers
            if 'dna_profile' in st.session_state.user_profile and 'blood_biomarkers' in st.session_state.user_profile[
                'dna_profile']:
                st.subheader("Your Blood Biomarkers")

                biomarkers = st.session_state.user_profile['dna_profile']['blood_biomarkers']
                for marker, value in biomarkers.items():
                    if marker == 'Cholesterol':
                        if value < 200:
                            st.success(f"{marker}: {value:.1f} mg/dL (Desirable)")
                        elif value < 240:
                            st.warning(f"{marker}: {value:.1f} mg/dL (Borderline high)")
                        else:
                            st.error(f"{marker}: {value:.1f} mg/dL (High)")
                    elif marker == 'HDL':
                        if value >= 60:
                            st.success(f"{marker}: {value:.1f} mg/dL (Optimal)")
                        elif value >= 40:
                            st.info(f"{marker}: {value:.1f} mg/dL (Normal)")
                        else:
                            st.warning(f"{marker}: {value:.1f} mg/dL (Low)")
                    elif marker == 'LDL':
                        if value < 100:
                            st.success(f"{marker}: {value:.1f} mg/dL (Optimal)")
                        elif value < 130:
                            st.info(f"{marker}: {value:.1f} mg/dL (Near optimal)")
                        elif value < 160:
                            st.warning(f"{marker}: {value:.1f} mg/dL (Borderline high)")
                        else:
                            st.error(f"{marker}: {value:.1f} mg/dL (High)")
                    elif marker == 'Glucose':
                        if value < 100:
                            st.success(f"{marker}: {value:.1f} mg/dL (Normal)")
                        elif value < 126:
                            st.warning(f"{marker}: {value:.1f} mg/dL (Prediabetes)")
                        else:
                            st.error(f"{marker}: {value:.1f} mg/dL (Diabetes range)")
                    else:
                        st.write(f"{marker}: {value:.1f}")

    with col2:
        if st.session_state.user_profile['dna_analyzed'] and 'dna_profile' in st.session_state.user_profile:
            # Display genetic variants
            if 'genetic_variants' in st.session_state.user_profile['dna_profile']:
                st.subheader("Your Genetic Variants")

                variants = st.session_state.user_profile['dna_profile']['genetic_variants']
                for gene, variant in variants.items():
                    with st.expander(f"{gene}: {variant}"):
                        if gene == 'BRCA1':
                            if variant == 'Present':
                                st.write(
                                    "The BRCA1 mutation is associated with increased risk of breast and ovarian cancer. Consider discussing screening options with your doctor.")
                            else:
                                st.write(
                                    "You do not have the common BRCA1 mutations associated with increased cancer risk.")
                        elif gene == 'MTHFR':
                            if variant == 'C677T Homozygous':
                                st.write(
                                    "This variant affects how your body processes folate and may impact homocysteine levels. Consider B-vitamin supplementation.")
                            elif variant == 'C677T Heterozygous':
                                st.write("This variant may have a mild effect on folate metabolism.")
                            else:
                                st.write("Your MTHFR gene is functioning normally.")
                        elif gene == 'ApoE':
                            if 'e4' in variant:
                                st.write(
                                    "This variant is associated with increased risk for Alzheimer's disease and cardiovascular issues. Focus on brain and heart health.")
                            elif 'e2' in variant:
                                st.write(
                                    "This variant is generally associated with lower cholesterol levels and lower risk of Alzheimer's.")
                            else:
                                st.write("This is the most common variant and is considered neutral for disease risk.")
                        else:
                            st.write(f"Information about {gene} variant: {variant}")

            # Display disease predispositions
            if 'disease_predispositions' in st.session_state.user_profile['dna_profile']:
                st.subheader("Disease Predisposition Analysis")

                # Create dataframe for visualization
                predispositions = st.session_state.user_profile['dna_profile']['disease_predispositions']

                # Get the top risks
                top_risks = dict(sorted(predispositions.items(), key=lambda item: item[1], reverse=True)[:10])

                # Create a bar chart
                fig = px.bar(
                    x=[risk.replace('_', ' ').title() for risk in top_risks.keys()],
                    y=[v * 100 for v in top_risks.values()],
                    labels={'x': 'Condition', 'y': 'Genetic Risk (%)'},
                    title='Top 10 Genetic Risk Factors'
                )

                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Show detailed risk information
                st.write("### Detailed Risk Analysis")
                for disease, risk in sorted(predispositions.items(), key=lambda item: item[1], reverse=True)[:5]:
                    formatted_disease = disease.replace('_', ' ').title()
                    risk_percentage = risk * 100

                    with st.expander(f"{formatted_disease}: {risk_percentage:.1f}% genetic risk"):
                        if risk_percentage > 20:
                            st.warning(f"Your genetic profile indicates an elevated risk for {formatted_disease}.")
                        elif risk_percentage > 10:
                            st.info(f"Your genetic profile shows a moderate risk for {formatted_disease}.")
                        else:
                            st.success(f"Your genetic risk for {formatted_disease} is relatively low.")

                        if disease in description_dict:
                            st.write(f"**About {formatted_disease}:**")
                            st.write(description_dict[disease])

                        if disease in precaution_dict:
                            st.write("**Preventive Measures:**")
                            for precaution in precaution_dict[disease]:
                                if precaution and precaution.strip():
                                    st.write(f"- {precaution}")
        else:
            st.subheader("Benefits of DNA Analysis")
            st.write("""
            Our DNA analysis provides:

            - **Disease Risk Assessment**: Identify genetic predispositions to various health conditions
            - **Personalized Prevention**: Get customized recommendations based on your genetic profile
            - **Medication Response**: Learn how your body may respond to certain medications
            - **Comprehensive Health Insights**: Integrate genetic data with symptoms and wearable data
            """)

            st.image("https://via.placeholder.com/400x300.png?text=DNA+Analysis", width=400)


def show_health_dashboard():
    st.title("Health Dashboard")
    st.write("Your comprehensive health overview")

    # Check if profile exists
    if not st.session_state.user_profile['name']:
        st.warning("Please set up your health profile to view your dashboard")
        return

    # Set up dashboard layout
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Vital Trends", "Health History", "Risk Assessment"])

    with tab1:
        # Quick glance at current health status
        st.subheader("Health Snapshot")

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.session_state.vital_trends:
                st.metric("Heart Rate", f"{st.session_state.vital_trends['heart_rate'][-1]} BPM")
            else:
                st.metric("Heart Rate", "-- BPM")

        with col2:
            if 'blood_pressure_sys' in st.session_state.vital_trends:
                bp_sys = st.session_state.vital_trends['blood_pressure_sys'][-1]
                bp_dia = st.session_state.vital_trends['blood_pressure_dia'][-1]
                st.metric("Blood Pressure", f"{bp_sys}/{bp_dia}")
            else:
                st.metric("Blood Pressure", "--/--")

        with col3:
            if st.session_state.vital_trends:
                st.metric("Steps Today", f"{st.session_state.vital_trends['steps'][-1]:,}")
            else:
                st.metric("Steps Today", "--")

        with col4:
            if 'sleep' in st.session_state.vital_trends:
                sleep = st.session_state.vital_trends['sleep'][-1]
                st.metric("Sleep", f"{sleep} hrs")
            else:
                st.metric("Sleep", "-- hrs")

        # Second row of insights
        st.subheader("Health Insights")

        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            st.write("**Activity Status**")

            if 'steps' in st.session_state.vital_trends:
                steps = st.session_state.vital_trends['steps'][-7:]
                avg_steps = sum(steps) / len(steps)

                if avg_steps < 5000:
                    st.warning(f"Your weekly average of {avg_steps:.0f} steps is below recommended levels.")
                elif avg_steps < 7500:
                    st.info(f"Your weekly average of {avg_steps:.0f} steps is moderate. Aim for 7,500+ steps daily.")
                else:
                    st.success(
                        f"Great job! Your weekly average of {avg_steps:.0f} steps meets activity recommendations.")
            else:
                st.info("Connect a wearable device to track your activity levels")

            # BMI status
            if st.session_state.user_profile['height'] > 0 and st.session_state.user_profile['weight'] > 0:
                bmi = st.session_state.user_profile['weight'] / ((st.session_state.user_profile['height'] / 100) ** 2)

                st.write("**BMI Status**")
                if bmi < 18.5:
                    st.warning(f"Your BMI is {bmi:.1f}, which is classified as underweight.")
                elif bmi < 25:
                    st.success(f"Your BMI is {bmi:.1f}, which is within the healthy range.")
                elif bmi < 30:
                    st.warning(f"Your BMI is {bmi:.1f}, which is classified as overweight.")
                else:
                    st.error(f"Your BMI is {bmi:.1f}, which is classified as obese.")

        with insight_col2:
            st.write("**Cardiovascular Health**")

            if 'heart_rate' in st.session_state.vital_trends and 'blood_pressure_sys' in st.session_state.vital_trends:
                hr = st.session_state.vital_trends['heart_rate'][-1]
                bp_sys = st.session_state.vital_trends['blood_pressure_sys'][-1]
                bp_dia = st.session_state.vital_trends['blood_pressure_dia'][-1]

                cardio_status = []

                # Heart rate status
                if hr < 60:
                    cardio_status.append("Your resting heart rate is low, which can be normal for athletes.")
                elif hr <= 100:
                    cardio_status.append("Your heart rate is within normal range.")
                else:
                    cardio_status.append("Your heart rate is elevated. Consider monitoring it closely.")

                # Blood pressure status
                if bp_sys < 120 and bp_dia < 80:
                    cardio_status.append("Your blood pressure is normal.")
                elif bp_sys < 130 and bp_dia < 80:
                    cardio_status.append("Your blood pressure is elevated.")
                elif bp_sys < 140 or bp_dia < 90:
                    cardio_status.append("Your blood pressure indicates hypertension stage 1.")
                else:
                    cardio_status.append(
                        "Your blood pressure indicates hypertension stage 2. Please consult a healthcare provider.")

                for status in cardio_status:
                    st.write(f"- {status}")
            else:
                st.info("Connect a heart rate and blood pressure monitor for cardiovascular insights")

            # Sleep status
            st.write("**Sleep Health**")
            if 'sleep' in st.session_state.vital_trends:
                sleep_data = st.session_state.vital_trends['sleep'][-7:]
                avg_sleep = sum(sleep_data) / len(sleep_data)

                if avg_sleep < 6:
                    st.error(f"Your average sleep of {avg_sleep:.1f} hours is insufficient. Aim for 7-9 hours.")
                elif avg_sleep < 7:
                    st.warning(f"Your average sleep of {avg_sleep:.1f} hours is slightly below recommendations.")
                elif avg_sleep <= 9:
                    st.success(f"Your average sleep of {avg_sleep:.1f} hours is within the recommended range.")
                else:
                    st.info(f"Your average sleep of {avg_sleep:.1f} hours is above typical recommendations.")
            else:
                st.info("Connect a sleep tracker for sleep health insights")

        # Third row: Recent health events and recommendations
        st.subheader("Recent Health Events")

        if st.session_state.health_history:
            # Get the most recent health event
            recent_event = st.session_state.health_history[-1]

            with st.expander(f"Health Analysis from {recent_event['date']} - View Details", expanded=True):
                st.write(f"**Symptoms:** {', '.join(recent_event['symptoms'])}")
                st.write(f"**Duration:** {recent_event['duration']} days")
                st.write(f"**Severity:** {recent_event['severity']}")

                st.write("**Potential Conditions:**")
                for disease, probability in recent_event['predictions']:
                    st.write(f"- {disease}: {probability}")

                if recent_event['recommendations']:
                    st.write("**Recommendations:**")
                    for rec in recent_event['recommendations']:
                        st.write(f"- {rec}")
        else:
            st.info("No health events recorded yet. Use the Symptom Analysis feature to log health events.")

        # Personalized health tips
        st.subheader("Personalized Health Tips")

        # Generate recommendations based on user profile
        tips = []

        # Activity recommendations
        if 'steps' in st.session_state.vital_trends:
            avg_steps = sum(st.session_state.vital_trends['steps'][-7:]) / 7
            if avg_steps < 5000:
                tips.append("Increase your daily activity by taking short walking breaks every hour")

        # Sleep recommendations
        if 'sleep' in st.session_state.vital_trends:
            avg_sleep = sum(st.session_state.vital_trends['sleep'][-7:]) / 7
            if avg_sleep < 7:
                tips.append(
                    "Improve your sleep quality by establishing a regular sleep schedule and avoiding screens before bedtime")

        # Lifestyle recommendations
        if st.session_state.user_profile['lifestyle']['exercise'] in ['Sedentary', 'Light']:
            tips.append("Consider incorporating more physical activity into your daily routine")

        if st.session_state.user_profile['lifestyle']['smoking'] in ['Occasionally', 'Regularly']:
            tips.append("Reducing or quitting smoking would significantly improve your overall health")

        # DNA-based recommendations
        if st.session_state.user_profile['dna_analyzed'] and 'genetic_variants' in st.session_state.user_profile[
            'dna_profile']:
            if st.session_state.user_profile['dna_profile']['genetic_variants'].get('ApoE', '').endswith('e4'):
                tips.append(
                    "Based on your ApoE variant, focus on cardiovascular health with a Mediterranean diet and regular exercise")

        # Display tips
        if tips:
            for tip in tips:
                st.info(tip)
        else:
            st.info("Connect more health data sources for personalized recommendations")

    with tab2:
        # Vital sign trends over time
        st.subheader("Vital Sign Trends")

        # Time period selection
        time_period = st.radio("Select Time Period", ["Week", "Month", "3 Months"], horizontal=True)

        if time_period == "Week":
            days_to_show = 7
        elif time_period == "Month":
            days_to_show = 30
        else:
            days_to_show = 90

        # Make sure we don't exceed available data
        max_days = len(st.session_state.vital_trends['dates'])
        days_to_show = min(days_to_show, max_days)

        # Select which vital signs to display
        vital_options = {
            "Heart Rate": "heart_rate",
            "Blood Pressure": "blood_pressure",
            "Temperature": "temperature",
            "Oxygen Saturation": "oxygen",
            "Sleep Duration": "sleep",
            "Steps": "steps"
        }

        selected_vitals = st.multiselect(
            "Select Vitals to Display",
            list(vital_options.keys()),
            default=["Heart Rate", "Steps"]
        )

        # Display selected vital sign charts
        for vital in selected_vitals:
            st.write(f"### {vital} Trend")

            vital_key = vital_options[vital]

            if vital_key == "blood_pressure":
                # Blood pressure needs special handling as it has systolic and diastolic values
                if 'blood_pressure_sys' in st.session_state.vital_trends and 'blood_pressure_dia' in st.session_state.vital_trends:
                    df = pd.DataFrame({
                        'Date': st.session_state.vital_trends['dates'][-days_to_show:],
                        'Systolic': st.session_state.vital_trends['blood_pressure_sys'][-days_to_show:],
                        'Diastolic': st.session_state.vital_trends['blood_pressure_dia'][-days_to_show:]
                    })

                    fig = px.line(
                        df, x='Date', y=['Systolic', 'Diastolic'],
                        labels={'value': 'mmHg', 'variable': 'Measurement'}
                    )

                    # Add reference lines for normal ranges
                    fig.add_hline(y=120, line_dash="dash", line_color="green", annotation_text="Normal Systolic")
                    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Normal Diastolic")

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Blood pressure data not available")
            else:
                # Regular vital signs with a single value
                if vital_key in st.session_state.vital_trends:
                    data = st.session_state.vital_trends[vital_key][-days_to_show:]
                    dates = st.session_state.vital_trends['dates'][-days_to_show:]

                    df = pd.DataFrame({
                        'Date': dates,
                        vital: data
                    })

                    fig = px.line(df, x='Date', y=vital)

                    # Add reference lines for normal ranges based on vital type
                    if vital_key == "heart_rate":
                        fig.add_hrect(y0=60, y1=100, line_width=0, fillcolor="green", opacity=0.2,
                                      annotation_text="Normal Range")
                    elif vital_key == "temperature":
                        fig.add_hrect(y0=36.1, y1=37.2, line_width=0, fillcolor="green", opacity=0.2,
                                      annotation_text="Normal Range")
                    elif vital_key == "oxygen":
                        fig.add_hrect(y0=95, y1=100, line_width=0, fillcolor="green", opacity=0.2,
                                      annotation_text="Normal Range")
                    elif vital_key == "sleep":
                        fig.add_hrect(y0=7, y1=9, line_width=0, fillcolor="green", opacity=0.2,
                                      annotation_text="Recommended Range")

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"{vital} data not available")

        # Correlations between vitals
        if len(selected_vitals) > 1:
            st.subheader("Vital Sign Correlations")

            # Create correlation dataframe from selected vitals
            corr_data = {}

            for vital in selected_vitals:
                vital_key = vital_options[vital]

                if vital_key == "blood_pressure":
                    if 'blood_pressure_sys' in st.session_state.vital_trends:
                        corr_data["Systolic BP"] = st.session_state.vital_trends['blood_pressure_sys'][-days_to_show:]
                    if 'blood_pressure_dia' in st.session_state.vital_trends:
                        corr_data["Diastolic BP"] = st.session_state.vital_trends['blood_pressure_dia'][-days_to_show:]
                elif vital_key in st.session_state.vital_trends:
                    corr_data[vital] = st.session_state.vital_trends[vital_key][-days_to_show:]

            if corr_data:
                corr_df = pd.DataFrame(corr_data)

                # Calculate correlations
                correlations = corr_df.corr()

                # Plot correlation heatmap
                fig = px.imshow(
                    correlations,
                    text_auto=True,
                    color_continuous_scale='Viridis',
                    title="Correlation Between Vital Signs"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Explain strongest correlations
                strongest_corr = []

                for i in range(len(correlations.columns)):
                    for j in range(i + 1, len(correlations.columns)):
                        col1 = correlations.columns[i]
                        col2 = correlations.columns[j]
                        corr_val = correlations.iloc[i, j]

                        if abs(corr_val) > 0.3:  # Only report meaningful correlations
                            strongest_corr.append((col1, col2, corr_val))

                if strongest_corr:
                    st.write("**Notable Correlations:**")

                    for col1, col2, corr_val in sorted(strongest_corr, key=lambda x: abs(x[2]), reverse=True):
                        if corr_val > 0.7:
                            relationship = "strong positive"
                        elif corr_val > 0.3:
                            relationship = "moderate positive"
                        elif corr_val > -0.3:
                            relationship = "weak"
                        elif corr_val > -0.7:
                            relationship = "moderate negative"
                        else:
                            relationship = "strong negative"

                        st.write(f"- {col1} and {col2}: {relationship} correlation ({corr_val:.2f})")

    with tab3:
        # Health history log
        st.subheader("Health History")

        if not st.session_state.health_history:
            st.info("No health events recorded yet. Use the Symptom Analysis feature to log health events.")
        else:
            # Create a dataframe from health history
            history_data = []

            for event in st.session_state.health_history:
                history_data.append({
                    'Date': event['date'],
                    'Symptoms': ", ".join(event['symptoms']),
                    'Duration': event['duration'],
                    'Severity': event['severity'],
                    'Top Prediction': event['predictions'][0][0] if event['predictions'] else "None",
                    'Probability': event['predictions'][0][1] if event['predictions'] else "N/A"
                })

            history_df = pd.DataFrame(history_data)

            # Display as a table
            st.dataframe(history_df, use_container_width=True)

            # Detailed view of selected event
            if len(st.session_state.health_history) > 0:
                event_dates = [event['date'] for event in st.session_state.health_history]
                selected_date = st.selectbox("View detailed health event", event_dates)

                selected_event = next(
                    (event for event in st.session_state.health_history if event['date'] == selected_date), None)

                if selected_event:
                    with st.expander("Health Event Details", expanded=True):
                        st.write(f"**Date:** {selected_event['date']}")
                        st.write(f"**Symptoms:** {', '.join(selected_event['symptoms'])}")
                        st.write(f"**Duration:** {selected_event['duration']} days")
                        st.write(f"**Severity:** {selected_event['severity']}")

                        st.write("**Potential Conditions:**")
                        for disease, probability in selected_event['predictions']:
                            st.write(f"- {disease}: {probability}")

                        if selected_event['recommendations']:
                            st.write("**Recommendations:**")
                            for rec in selected_event['recommendations']:
                                st.write(f"- {rec}")

            # Export options
            if st.button("Export Health History"):
                # In a real app, this would generate a downloadable report
                st.success("Health history report generated")

    with tab4:
        # Risk assessment and prediction
        st.subheader("Health Risk Assessment")

        # Combined risk factors
        risk_factors = []

        # Add lifestyle risk factors
        if st.session_state.user_profile['lifestyle']['smoking'] in ['Occasionally', 'Regularly']:
            risk_factors.append(("Smoking", "High", "Increases risk of lung cancer, heart disease, and stroke"))

        if st.session_state.user_profile['lifestyle']['alcohol'] in ['Moderate', 'Regular']:
            risk_factors.append(
                ("Alcohol consumption", "Moderate", "May impact liver health and increase various cancer risks"))

        if st.session_state.user_profile['lifestyle']['exercise'] in ['Sedentary', 'Light']:
            risk_factors.append(
                ("Low physical activity", "Moderate", "Increases risk of cardiovascular disease and diabetes"))

        # Add BMI-related risk
        if st.session_state.user_profile['height'] > 0 and st.session_state.user_profile['weight'] > 0:
            bmi = st.session_state.user_profile['weight'] / ((st.session_state.user_profile['height'] / 100) ** 2)

            if bmi > 30:
                risk_factors.append(
                    ("Obesity", "High", "Increases risk of diabetes, heart disease, and certain cancers"))
            elif bmi > 25:
                risk_factors.append(("Overweight", "Moderate", "May increase risk of heart disease and diabetes"))
            elif bmi < 18.5:
                risk_factors.append(
                    ("Underweight", "Moderate", "May indicate nutritional deficiencies or other health concerns"))

        # Add blood pressure related risk
        if 'blood_pressure_sys' in st.session_state.vital_trends and 'blood_pressure_dia' in st.session_state.vital_trends:
            bp_sys = st.session_state.vital_trends['blood_pressure_sys'][-1]
            bp_dia = st.session_state.vital_trends['blood_pressure_dia'][-1]

            if bp_sys >= 140 or bp_dia >= 90:
                risk_factors.append(
                    ("Hypertension", "High", "Increases risk of heart disease, stroke, and kidney damage"))
            elif bp_sys >= 130 or bp_dia >= 80:
                risk_factors.append(
                    ("Elevated Blood Pressure", "Moderate", "May develop into hypertension if not managed"))

        # Add genetic risk factors
        if st.session_state.user_profile['dna_analyzed'] and 'disease_predispositions' in st.session_state.user_profile[
            'dna_profile']:
            for disease, risk in sorted(st.session_state.user_profile['dna_profile']['disease_predispositions'].items(),
                                        key=lambda x: x[1], reverse=True)[:3]:
                if risk > 0.2:
                    risk_level = "High"
                elif risk > 0.1:
                    risk_level = "Moderate"
                else:
                    risk_level = "Low"

                formatted_disease = disease.replace('_', ' ').title()
                risk_factors.append((f"Genetic risk for {formatted_disease}", risk_level,
                                     f"Genetic predisposition with {risk * 100:.1f}% risk factor"))

        # Display risk factors
        if risk_factors:
            risk_df = pd.DataFrame(risk_factors, columns=["Risk Factor", "Risk Level", "Description"])

            # Color-code risk levels
            def color_risk_level(val):
                if val == "High":
                    return 'background-color: #ffcccc'
                elif val == "Moderate":
                    return 'background-color: #ffffcc'
                else:
                    return 'background-color: #ccffcc'

            styled_risk_df = risk_df.style.applymap(color_risk_level, subset=['Risk Level'])

            st.dataframe(styled_risk_df, use_container_width=True)
        else:
            st.info("No significant risk factors identified. Continue maintaining your healthy lifestyle.")

        # Future disease prediction
        st.subheader("Predictive Health Insights")

        # Generate predictive insights based on user data
        if st.session_state.user_profile['dna_analyzed']:
            st.write("### Based on your genetic profile and lifestyle:")

            # Combine genetic and lifestyle factors to generate predictions
            predictions = []

            if 'disease_predispositions' in st.session_state.user_profile['dna_profile']:
                genetic_risks = st.session_state.user_profile['dna_profile']['disease_predispositions']

                # Get top genetic risks
                top_genetic_risks = dict(sorted(genetic_risks.items(), key=lambda item: item[1], reverse=True)[:5])

                # Modify risks based on lifestyle
                for disease, base_risk in top_genetic_risks.items():
                    modified_risk = base_risk
                    risk_factors = []
                    protective_factors = []

                    # Apply lifestyle modifiers
                    if disease.lower() in ['heart_disease', 'hypertension', 'stroke', 'coronary_artery_disease']:
                        if st.session_state.user_profile['lifestyle']['smoking'] in ['Occasionally', 'Regularly']:
                            modified_risk *= 1.5
                            risk_factors.append("smoking")

                        if st.session_state.user_profile['lifestyle']['exercise'] in ['Active', 'Very Active']:
                            modified_risk *= 0.7
                            protective_factors.append("regular exercise")

                        if st.session_state.user_profile['lifestyle']['diet'] == 'Mediterranean':
                            modified_risk *= 0.8
                            protective_factors.append("Mediterranean diet")

                    elif disease.lower() in ['diabetes', 'type_2_diabetes']:
                        if st.session_state.user_profile['height'] > 0 and st.session_state.user_profile['weight'] > 0:
                            bmi = st.session_state.user_profile['weight'] / (
                                        (st.session_state.user_profile['height'] / 100) ** 2)
                            if bmi > 30:
                                modified_risk *= 2.0
                                risk_factors.append("obesity")

                        if st.session_state.user_profile['lifestyle']['exercise'] in ['Active', 'Very Active']:
                            modified_risk *= 0.6
                            protective_factors.append("regular exercise")

                    elif disease.lower() in ['lung_cancer', 'copd', 'emphysema']:
                        if st.session_state.user_profile['lifestyle']['smoking'] in ['Occasionally', 'Regularly']:
                            modified_risk *= 3.0
                            risk_factors.append("smoking")

                    # Add to predictions list
                    formatted_disease = disease.replace('_', ' ').title()

                    prediction_text = f"Your risk for {formatted_disease} is "
                    if modified_risk > 0.5:
                        prediction_text += "high"
                    elif modified_risk > 0.2:
                        prediction_text += "moderate"
                    else:
                        prediction_text += "relatively low"

                    if risk_factors:
                        prediction_text += f". Risk is increased by: {', '.join(risk_factors)}"

                    if protective_factors:
                        prediction_text += f". Risk is decreased by: {', '.join(protective_factors)}"

                    predictions.append((formatted_disease, modified_risk, prediction_text))

                # Display top 3 predictions
                for i, (disease, risk, text) in enumerate(sorted(predictions, key=lambda x: x[1], reverse=True)[:3]):
                    expander_label = f"{disease}: {'High' if risk > 0.5 else 'Moderate' if risk > 0.2 else 'Low'} Risk"

                    with st.expander(expander_label):
                        st.write(text)

                        if disease.lower().replace(' ', '_') in precaution_dict:
                            st.write("**Preventive Measures:**")
                            for precaution in precaution_dict[disease.lower().replace(' ', '_')]:
                                if precaution and precaution.strip():
                                    st.write(f"- {precaution}")
        else:
            st.info("Add DNA analysis for personalized predictive health insights")


def show_predictive_insights():
    st.title("Predictive Health Insights")
    st.write("AI-powered analysis for proactive health management")

    # Check if profile exists
    if not st.session_state.user_profile['name']:
        st.warning("Please set up your health profile to access predictive insights")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Health Status")

        # Get health score based on available data
        health_factors = []
        health_score = 70  # Base score

        # Lifestyle factors
        if st.session_state.user_profile['lifestyle']['exercise'] == 'Very Active':
            health_score += 10
            health_factors.append(("Exercise", "Very Active", "+10"))
        elif st.session_state.user_profile['lifestyle']['exercise'] == 'Active':
            health_score += 5
            health_factors.append(("Exercise", "Active", "+5"))
        elif st.session_state.user_profile['lifestyle']['exercise'] == 'Sedentary':
            health_score -= 5
            health_factors.append(("Exercise", "Sedentary", "-5"))

        if st.session_state.user_profile['lifestyle']['smoking'] == 'Regularly':
            health_score -= 15
            health_factors.append(("Smoking", "Regular", "-15"))
        elif st.session_state.user_profile['lifestyle']['smoking'] == 'Occasionally':
            health_score -= 7
            health_factors.append(("Smoking", "Occasional", "-7"))

        if st.session_state.user_profile['lifestyle']['alcohol'] == 'Regular':
            health_score -= 8
            health_factors.append(("Alcohol", "Regular", "-8"))

        # BMI
        if st.session_state.user_profile['height'] > 0 and st.session_state.user_profile['weight'] > 0:
            bmi = st.session_state.user_profile['weight'] / ((st.session_state.user_profile['height'] / 100) ** 2)

            if 18.5 <= bmi < 25:
                health_score += 5
                health_factors.append(("BMI", f"{bmi:.1f} (Normal)", "+5"))
            elif 25 <= bmi < 30:
                health_score -= 3
                health_factors.append(("BMI", f"{bmi:.1f} (Overweight)", "-3"))
            elif bmi >= 30:
                health_score -= 10
                health_factors.append(("BMI", f"{bmi:.1f} (Obese)", "-10"))
            else:
                health_score -= 5
                health_factors.append(("BMI", f"{bmi:.1f} (Underweight)", "-5"))

        # Sleep
        if 'sleep' in st.session_state.vital_trends:
            avg_sleep = sum(st.session_state.vital_trends['sleep'][-7:]) / 7

            if 7 <= avg_sleep <= 9:
                health_score += 5
                health_factors.append(("Sleep", f"{avg_sleep:.1f} hrs (Optimal)", "+5"))
            elif 6 <= avg_sleep < 7:
                health_score -= 2
                health_factors.append(("Sleep", f"{avg_sleep:.1f} hrs (Suboptimal)", "-2"))
            elif avg_sleep < 6:
                health_score -= 8
                health_factors.append(("Sleep", f"{avg_sleep:.1f} hrs (Insufficient)", "-8"))

        # Activity
        if 'steps' in st.session_state.vital_trends:
            avg_steps = sum(st.session_state.vital_trends['steps'][-7:]) / 7

            if avg_steps >= 10000:
                health_score += 8
                health_factors.append(("Daily Steps", f"{avg_steps:.0f} (Excellent)", "+8"))
            elif avg_steps >= 7500:
                health_score += 5
                health_factors.append(("Daily Steps", f"{avg_steps:.0f} (Good)", "+5"))
            elif avg_steps < 5000:
                health_score -= 5
                health_factors.append(("Daily Steps", f"{avg_steps:.0f} (Low)", "-5"))

        # Blood pressure
        if 'blood_pressure_sys' in st.session_state.vital_trends:
            bp_sys = st.session_state.vital_trends['blood_pressure_sys'][-1]
            bp_dia = st.session_state.vital_trends['blood_pressure_dia'][-1]

            if bp_sys < 120 and bp_dia < 80:
                health_score += 5
                health_factors.append(("Blood Pressure", f"{bp_sys}/{bp_dia} (Normal)", "+5"))
            elif bp_sys < 130 and bp_dia < 80:
                health_score += 0
                health_factors.append(("Blood Pressure", f"{bp_sys}/{bp_dia} (Elevated)", "0"))
            elif bp_sys < 140 or bp_dia < 90:
                health_score -= 5
                health_factors.append(("Blood Pressure", f"{bp_sys}/{bp_dia} (Stage 1 Hypertension)", "-5"))
            else:
                health_score -= 10
                health_factors.append(("Blood Pressure", f"{bp_sys}/{bp_dia} (Stage 2 Hypertension)", "-10"))

        # Cap score between 0 and 100
        health_score = max(0, min(100, health_score))

        # Display health score
        st.write(f"### Your Health Score: {health_score}/100")

        # Create gauge chart for health score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Health Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': health_score
                }
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # Display health factors
        if health_factors:
            st.write("### Health Score Factors")

            factors_df = pd.DataFrame(health_factors, columns=["Factor", "Status", "Impact"])

            # Color-code impact
            def color_impact(val):
                if val.startswith("+"):
                    return 'color: green'
                elif val.startswith("-"):
                    return 'color: red'
                else:
                    return 'color: black'

            styled_factors_df = factors_df.style.applymap(color_impact, subset=['Impact'])

            st.dataframe(styled_factors_df, use_container_width=True)

    with col2:
        st.subheader("Predictive Alerts")

        # Generate alerts based on health data
        alerts = []

        # Check for concerning trends
        if 'heart_rate' in st.session_state.vital_trends:
            hr_data = st.session_state.vital_trends['heart_rate'][-7:]
            avg_hr = sum(hr_data) / len(hr_data)

            if avg_hr > 90:
                alerts.append(("Elevated Heart Rate", "Warning",
                               f"Your average heart rate of {avg_hr:.0f} BPM is higher than normal. Consider stress reduction techniques and consult a doctor if it persists."))

        if 'blood_pressure_sys' in st.session_state.vital_trends:
            bp_sys_data = st.session_state.vital_trends['blood_pressure_sys'][-7:]
            bp_dia_data = st.session_state.vital_trends['blood_pressure_dia'][-7:]

            avg_sys = sum(bp_sys_data) / len(bp_sys_data)
            avg_dia = sum(bp_dia_data) / len(bp_dia_data)

            if avg_sys >= 140 or avg_dia >= 90:
                alerts.append(("Hypertension Risk", "High",
                               f"Your average blood pressure of {avg_sys:.0f}/{avg_dia:.0f} mmHg indicates hypertension. Please consult a healthcare provider."))
            elif avg_sys >= 130 or avg_dia >= 85:
                alerts.append(("Elevated Blood Pressure", "Moderate",
                               f"Your average blood pressure of {avg_sys:.0f}/{avg_dia:.0f} mmHg is elevated. Consider monitoring and lifestyle modifications."))

        if 'sleep' in st.session_state.vital_trends:
            sleep_data = st.session_state.vital_trends['sleep'][-7:]
            avg_sleep = sum(sleep_data) / len(sleep_data)

            if avg_sleep < 6:
                alerts.append(("Sleep Deficiency", "High",
                               f"Your average sleep of {avg_sleep:.1f} hours is significantly below recommendations. Chronic sleep deficiency increases risk of several health conditions."))
            elif avg_sleep < 7:
                alerts.append(("Suboptimal Sleep", "Moderate",
                               f"Your average sleep of {avg_sleep:.1f} hours is below the recommended 7-9 hours. Consider adjusting your sleep schedule."))

        # Check for weight changes if tracked over time
        if st.session_state.health_history and len(st.session_state.health_history) > 1:
            recent_symptoms = [event['symptoms'] for event in st.session_state.health_history[-3:]]
            flat_symptoms = [item for sublist in recent_symptoms for item in sublist]

            # Check for recurring symptoms
            symptom_counts = {}
            for symptom in flat_symptoms:
                symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1

            recurring_symptoms = [s for s, count in symptom_counts.items() if count >= 2]

            if recurring_symptoms:
                alerts.append(("Recurring Symptoms", "Moderate",
                               f"You've reported these symptoms multiple times: {', '.join(recurring_symptoms)}. Consider discussing with a healthcare provider."))

        # DNA-based alerts
        if st.session_state.user_profile['dna_analyzed'] and 'disease_predispositions' in st.session_state.user_profile[
            'dna_profile']:
            high_risk_diseases = [(disease, risk) for disease, risk in
                                  st.session_state.user_profile['dna_profile']['disease_predispositions'].items() if
                                  risk > 0.3]

            for disease, risk in high_risk_diseases[:2]:  # Show top 2 high-risk alerts
                formatted_disease = disease.replace('_', ' ').title()
                alerts.append((f"{formatted_disease} Risk", "Genetic Predisposition",
                               f"Your genetic profile indicates a {risk * 100:.1f}% risk factor for {formatted_disease}. Consider preventive screenings."))

        # Display alerts
        if alerts:
            for severity, alert_group in {
                "High": [a for a in alerts if a[1] == "High" or a[1] == "Warning"],
                "Moderate": [a for a in alerts if a[1] == "Moderate"],
                "Genetic": [a for a in alerts if a[1] == "Genetic Predisposition"]
            }.items():
                if alert_group:
                    for title, level, message in alert_group:
                        if severity == "High":
                            st.error(f"**{title}**: {message}")
                        elif severity == "Moderate":
                            st.warning(f"**{title}**: {message}")
                        else:
                            st.info(f"**{title}**: {message}")
        else:
            st.success("No health alerts detected at this time. Continue maintaining your healthy habits!")

        # Upcoming preventive screenings
        st.subheader("Recommended Screenings")

        age = st.session_state.user_profile['age']
        gender = st.session_state.user_profile['gender']

        screenings = []

        # Age-based screening recommendations
        if gender == "Female":
            if age >= 40:
                screenings.append(("Mammogram", "Every 1-2 years"))
            if age >= 21:
                screenings.append(("Pap smear", "Every 3 years"))
            if age >= 45:
                screenings.append(("Colorectal cancer screening", "Starting at age 45"))
            if age >= 65:
                screenings.append(("Bone density test", "As recommended by doctor"))
        else:
            if age >= 45:
                screenings.append(("Colorectal cancer screening", "Starting at age 45"))
            if age >= 50:
                screenings.append(("Prostate cancer screening", "Discuss with doctor"))

        # Common screenings for all genders
        if age >= 18:
            screenings.append(("Blood pressure check", "Annually"))
        if age >= 20:
            screenings.append(("Cholesterol test", "Every 4-6 years"))
        if age >= 40:
            screenings.append(("Diabetes screening", "Every 3 years"))

        # Add additional screenings based on risk factors
        if st.session_state.user_profile['dna_analyzed'] and 'disease_predispositions' in st.session_state.user_profile[
            'dna_profile']:
            predispositions = st.session_state.user_profile['dna_profile']['disease_predispositions']

            if 'heart_disease' in predispositions and predispositions['heart_disease'] > 0.2:
                screenings.append(("Cardiac risk assessment", "Discuss frequency with doctor"))

            if 'skin_cancer' in predispositions and predispositions['skin_cancer'] > 0.2:
                screenings.append(("Skin cancer screening", "Annual skin checks"))

            if 'colon_cancer' in predispositions and predispositions['colon_cancer'] > 0.2:
                screenings.append(("Colonoscopy", "Earlier and more frequent than standard"))

        if screenings:
            screenings_df = pd.DataFrame(screenings, columns=["Screening", "Recommendation"])
            st.dataframe(screenings_df, use_container_width=True)
        else:
            st.info("No screenings recommended at this time based on your age and gender")


# Run the application
if __name__ == "__main__":
    main()
