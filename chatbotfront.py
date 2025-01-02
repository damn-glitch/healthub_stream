from flask import Flask, request, jsonify
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import csv

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

# Load data and preprocess
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = le.transform(testing['prognosis'])

clf = DecisionTreeClassifier().fit(x_train, y_train)

severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {symptom: index for index, symptom in enumerate(x)}


def load_dictionaries():
    global severityDictionary, description_list, precautionDictionary
    try:
        with open('symptom_Description.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) < 2:
                    print(f"Skipping malformed row in symptom_Description.csv: {row}")
                    continue
                description_list[row[0]] = row[1]

        with open('symptom_severity.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) < 2:
                    print(f"Skipping malformed row in symptom_severity.csv: {row}")
                    continue
                severityDictionary[row[0]] = int(row[1])

        with open('symptom_precaution.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) < 5:
                    print(f"Skipping malformed row in symptom_precaution.csv: {row}")
                    continue
                precautionDictionary[row[0]] = row[1:]

    except Exception as e:
        print(f"Error loading dictionaries: {e}")



load_dictionaries()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', [])
    num_days = data.get('days', 1)

    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms:
        input_vector[symptoms_dict[symptom]] = 1

    first_prediction = clf.predict([input_vector])[0]
    present_disease = le.inverse_transform([first_prediction])[0]

    symptoms_given = list(reduced_data.columns[reduced_data.loc[present_disease].values[0].nonzero()])
    symptoms_exp = [symptom for symptom in symptoms_given if symptom in symptoms]

    def calc_condition(exp, days):
        sum_severity = sum(severityDictionary[item] for item in exp)
        return "consultation from doctor" if (sum_severity * days) / (len(exp) + 1) > 13 else "precautions"

    condition = calc_condition(symptoms_exp, num_days)

    response = {
        "disease": present_disease,
        "description": description_list[present_disease],
        "precautions": precautionDictionary[present_disease],
        "condition": condition
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
