import pandas as pd
import numpy as np
import streamlit as st
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load data
@st.cache_data
def load_data():
    training = pd.read_csv('Training.csv')
    testing = pd.read_csv('Testing.csv')
    severity_df = pd.read_csv('symptom_severity.csv')
    description_df = pd.read_csv('symptom_Description.csv')
    precaution_df = pd.read_csv('symptom_precaution.csv')
    return training, testing, severity_df, description_df, precaution_df

training, testing, severity_df, description_df, precaution_df = load_data()

severity_dict = dict(zip(severity_df.iloc[:, 0], severity_df.iloc[:, 1]))
description_dict = dict(zip(description_df.iloc[:, 0], description_df.iloc[:, 1]))
precaution_dict = dict(zip(precaution_df.iloc[:, 0], precaution_df.iloc[:, 1:].values.tolist()))

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

# Train and evaluate SVM
model = SVC()
model.fit(x_train, y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

st.title("Health Diagnosis System")
st.write("Enter your symptoms and get a probable diagnosis.")

symptoms_dict = {symptom: index for index, symptom in enumerate(x)}

def calc_condition(exp, days):
    sum_severity = sum(severity_dict[item] for item in exp)
    if (sum_severity * days) / (len(exp) + 1) > 13:
        st.write("You should take consultation from a doctor.")
    else:
        st.write("It might not be that bad but you should take precautions.")

def sec_predict(symptoms_exp):
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1
    return clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return disease

def tree_to_code(tree, feature_names, disease_input, num_days):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            st.session_state.symptoms_given = symptoms_given.tolist()
            st.session_state.present_disease = present_disease[0]
            st.session_state.step = 'additional_symptoms'
            st.session_state.symptoms_exp = []
            st.session_state.current_symptom_index = 0

    recurse(0, 1)

def get_additional_symptoms(symptoms_given, current_symptom_index):
    # Display previously answered symptoms
    for i in range(current_symptom_index):
        syms = symptoms_given[i].replace('_', ' ')
        inp = st.session_state.symptoms_exp[i]
        st.write(f"{syms}: {inp}")

    # Display the current symptom question
    if current_symptom_index < len(symptoms_given):
        syms = symptoms_given[current_symptom_index].replace('_', ' ')
        inp = st.radio(f"Are you experiencing {syms}?", ("Yes", "No"), key=syms)
        if st.button("Next Symptom"):
            st.session_state.symptoms_exp.append(inp)
            st.session_state.current_symptom_index += 1
            st.experimental_rerun()

# Initial user input
user_name = st.text_input("Enter your name")
symptom_input = st.text_input("Enter the symptom you are experiencing")

if user_name and symptom_input and 'step' not in st.session_state:
    symptom_matches = [symptom for symptom in symptoms_dict if symptom_input.lower() in symptom.lower()]
    if symptom_matches:
        selected_symptom = st.selectbox("Select the one you meant", [symptom.replace('_', ' ') for symptom in symptom_matches])
        num_days = st.number_input("For how many days have you been experiencing this?", min_value=1, max_value=30, step=1)

        if st.button("Next"):
            st.session_state.selected_symptom = selected_symptom.replace(' ', '_')
            st.session_state.num_days = num_days
            st.session_state.step = 'diagnose'
            tree_to_code(clf, cols, st.session_state.selected_symptom, num_days)

# Additional symptoms and diagnosis
if 'step' in st.session_state and st.session_state.step == 'additional_symptoms':
    current_symptom_index = st.session_state.get('current_symptom_index', 0)

    get_additional_symptoms(st.session_state.symptoms_given, current_symptom_index)

    if current_symptom_index >= len(st.session_state.symptoms_given):
        symptoms_exp = [symptom for symptom, response in zip(st.session_state.symptoms_given, st.session_state.symptoms_exp) if response == "Yes"]
        second_prediction = sec_predict(symptoms_exp)
        second_prediction_disease = le.inverse_transform(second_prediction)[0]
        calc_condition(symptoms_exp, st.session_state.num_days)
        if st.session_state.present_disease == second_prediction_disease:
            st.write(f"You may have {st.session_state.present_disease.replace('_', ' ')}")
            if st.session_state.present_disease in description_dict:
                st.write(description_dict[st.session_state.present_disease])
        else:
            st.write(f"You may have {st.session_state.present_disease.replace('_', ' ')} or {second_prediction_disease.replace('_', ' ')}")
            if st.session_state.present_disease in description_dict:
                st.write(description_dict[st.session_state.present_disease])
            if second_prediction_disease in description_dict:
                st.write(description_dict[second_prediction_disease])

        if st.session_state.present_disease in precaution_dict:
            precautions = precaution_dict[st.session_state.present_disease]
            st.write("Take the following measures: ")
            for i, prec in enumerate(precautions):
                st.write(f"{i + 1}. {prec}")

