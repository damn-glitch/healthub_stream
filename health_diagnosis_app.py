import streamlit as st
import pandas as pd
import numpy as np
import csv
import pyttsx3
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=DeprecationWarning)


########################
#      LOAD DATA       #
########################
@st.cache_data
def load_data():
    training = pd.read_csv('Training.csv')
    testing = pd.read_csv('Testing.csv')

    # Prepare columns
    cols = training.columns
    cols = cols[:-1]

    # X, y for training
    x = training[cols]
    y = training['prognosis']

    # Encode labels
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    # X, y for testing
    x_test = testing[cols]
    y_test = testing['prognosis']
    y_test = le.transform(y_test)

    # Train-test split
    x_train, x_val, y_train, y_val = train_test_split(x, y_encoded,
                                                      test_size=0.33,
                                                      random_state=42)
    return {
        "training": training,
        "testing": testing,
        "cols": cols,
        "X_train": x_train,
        "X_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_test": x_test,
        "y_test": y_test,
        "le": le
    }


@st.cache_data
def load_dictionary_files():
    """
    Loads:
      - symptom_Description.csv -> description_list
      - symptom_severity.csv -> severityDictionary
      - symptom_precaution.csv -> precautionDictionary
    """
    # symptom_Description
    description_list = {}
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

    # symptom_severity
    severityDictionary = {}
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            try:
                severityDictionary[row[0]] = int(row[1])
            except:
                pass

    # symptom_precaution
    precautionDictionary = {}
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

    return description_list, severityDictionary, precautionDictionary


########################
#    BUILD MODELS      #
########################
@st.cache_resource
def build_models(x_train, y_train):
    """
    Build and return two models:
      - Decision Tree
      - SVM
    """
    # Decision Tree
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    # SVM
    model_svc = SVC()
    model_svc.fit(x_train, y_train)

    return clf, model_svc


########################
#    PREDICTION LOGIC  #
########################
def second_prediction(symptoms_list, cols, training):
    """
    Uses a Decision Tree trained on the entire Training.csv
    for a secondary check on potential disease.
    """
    # Re-train a minimal DT for secondary prediction:
    X = training.iloc[:, :-1]
    y = training['prognosis']

    dt2 = DecisionTreeClassifier()
    dt2.fit(X, y)

    # Create an input vector from the selected symptoms
    all_symptoms = list(cols)  # all symptom columns
    input_vector = np.zeros(len(all_symptoms))

    for sym in symptoms_list:
        if sym in all_symptoms:
            idx = all_symptoms.index(sym)
            input_vector[idx] = 1

    return dt2.predict([input_vector])


def calc_condition(symptoms_exp, days, severity_dict):
    """
    Simple threshold-based rule: if the average severity
    * days is above 13, see a doctor.
    """
    total_severity = 0
    for symptom in symptoms_exp:
        total_severity += severity_dict.get(symptom, 0)
    if (total_severity * days) / (len(symptoms_exp) + 1) > 13:
        return "You should consult a doctor."
    else:
        return "It might not be that bad but still take precautions."


def text_to_speech(text):
    """Optionally use pyttsx3 (if desired)."""
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


########################
#    STREAMLIT UI      #
########################
def main():
    # ============== PAGE CONFIG ==============
    st.set_page_config(page_title="Disease Prediction", layout="wide")

    # ============== LOAD DATA ================
    data_dict = load_data()
    training_data = data_dict["training"]
    cols = data_dict["cols"]
    le = data_dict["le"]

    description_dict, severity_dict, precaution_dict = load_dictionary_files()

    # ============== BUILD MODELS =============
    clf, svc_model = build_models(data_dict["X_train"], data_dict["y_train"])

    # ============== SIDEBAR ==================
    st.sidebar.title("Disease Prediction")
    st.sidebar.markdown(
        """
        This app predicts potential diseases based on your symptoms.\n
        **Instructions**:
        - Select all your symptoms.
        - Enter how many days you have had these symptoms.
        - Click *Predict Disease*.
        """
    )
    # Let user select multiple symptoms
    selected_symptoms = st.sidebar.multiselect(
        "Select your symptoms",
        options=cols,
        default=[]
    )

    # Number of days input
    days = st.sidebar.number_input("Number of days with these symptoms", min_value=1, value=1)

    # ============== MAIN CONTENT =============
    st.title("Disease Prediction System")
    st.markdown(
        """
        <style>
        .stMarkdown p {
            font-size:18px !important;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown("Please fill in the information in the sidebar, then click **Predict Disease**.")

    # Predict disease button
    if st.sidebar.button("Predict Disease"):
        # Convert selected symptoms to a 1/0 input for the model
        input_vector = np.zeros(len(cols))
        for symptom in selected_symptoms:
            idx = list(cols).index(symptom)
            input_vector[idx] = 1

        # Predict with Decision Tree (primary)
        primary_pred_idx = clf.predict([input_vector])[0]  # numeric
        primary_disease = le.inverse_transform([primary_pred_idx])[0]

        # Predict with the "second" model for cross-check
        secondary_disease = second_prediction(selected_symptoms, cols, training_data)[0]

        # Condition severity
        condition_msg = calc_condition(selected_symptoms, days, severity_dict)

        # OUTPUT RESULTS
        st.subheader("Prediction Results")
        st.write(f"**Primary Prediction**: {primary_disease}")
        st.write(f"**Possible Alternative**: {secondary_disease}")
        st.write(f"**Advice**: {condition_msg}")

        # Show disease descriptions
        st.subheader("Disease Descriptions")
        if primary_disease in description_dict:
            st.markdown(f"**{primary_disease}**: {description_dict[primary_disease]}")
        if secondary_disease in description_dict and secondary_disease != primary_disease:
            st.markdown(f"**{secondary_disease}**: {description_dict[secondary_disease]}")

        # Precautions
        st.subheader("Suggested Precautions")
        if primary_disease in precaution_dict:
            st.markdown(f"**For {primary_disease}**:")
            for i, prec in enumerate(precaution_dict[primary_disease], start=1):
                st.write(f"{i}. {prec}")

        if (secondary_disease != primary_disease) and (secondary_disease in precaution_dict):
            st.markdown(f"**For {secondary_disease}**:")
            for i, prec in enumerate(precaution_dict[secondary_disease], start=1):
                st.write(f"{i}. {prec}")

        # Optional: text to speech
        # text_to_speech(f"You may have {primary_disease} or {secondary_disease}")

    # Show some model info
    with st.expander("Model Performance"):
        # Evaluate Decision Tree
        dt_cv_scores = cross_val_score(clf, data_dict["X_val"], data_dict["y_val"], cv=3).mean()
        st.write(f"**Decision Tree Accuracy (CV)**: {round(dt_cv_scores * 100, 2)}%")

        # Evaluate SVC
        svc_acc = svc_model.score(data_dict["X_val"], data_dict["y_val"])
        st.write(f"**SVM Accuracy (Validation)**: {round(svc_acc * 100, 2)}%")

    st.write("---")


if __name__ == "__main__":
    main()
