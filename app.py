from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import warnings
import csv

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

# Load datasets
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')

# Prepare data
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
reduced_data = training.groupby(training['prognosis']).max()

# Encode labels
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Train model
clf = DecisionTreeClassifier().fit(x, y)

# Initialize dictionaries
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()
symptoms_dict = {symptom: index for index, symptom in enumerate(x.columns)}

def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) >= 2:
                description_list[row[0]] = row[1]

def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) >= 2:
                try:
                    symptom = row[0].strip()
                    severity = int(row[1].strip())
                    severityDictionary[symptom] = severity
                except ValueError:
                    continue

def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) >= 5:
                precautionDictionary[row[0]] = row[1:]

def calc_condition(exp, days):
    sum_severity = sum(severityDictionary.get(item, 0) for item in exp)
    if (sum_severity * days) / (len(exp) + 1) > 13:
        return "You should consult a doctor."
    else:
        return "It might not be that bad, but you should take precautions."

def predictDisease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        index = symptoms_dict.get(symptom)
        if index is not None:
            input_vector[index] = 1
    prediction = clf.predict([input_vector])
    disease = le.inverse_transform(prediction)[0]
    return disease

# Load data
getSeverityDict()
getDescription()
getprecautionDict()

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    symptoms_list = sorted(symptoms_dict.keys())
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        num_days = int(request.form['days']) if request.form['days'] else 0

        if not selected_symptoms or not num_days:
            error_message = "Please select symptoms and enter the number of days."
            return render_template('index.html', symptoms_list=symptoms_list, error_message=error_message)

        disease = predictDisease(selected_symptoms)
        condition = calc_condition(selected_symptoms, num_days)
        description = description_list.get(disease, "No description available.")
        precautions = precautionDictionary.get(disease, [])

        return render_template('result.html', disease=disease, condition=condition,
                               description=description, precautions=precautions)

    return render_template('index.html', symptoms_list=symptoms_list)

if __name__ == '__main__':
    app.run(debug=True)
