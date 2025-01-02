import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import tkinter as tk
from tkinter import messagebox, ttk

warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = le.transform(testing['prognosis'])

# Train models
clf = DecisionTreeClassifier().fit(x_train, y_train)
model = SVC().fit(x_train, y_train)

# Initialize dictionaries
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()
symptoms_dict = {symptom: index for index, symptom in enumerate(x)}

def getDescription():
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            description_list[row[0]] = row[1]

def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            # Check if the row has at least two columns
            if len(row) >= 2:
                try:
                    symptom = row[0].strip()
                    severity = int(row[1].strip())
                    severityDictionary[symptom] = severity
                except ValueError:
                    # Handle the case where severity is not an integer
                    print(f"Invalid severity value for symptom '{symptom}': {row[1]}")
            else:
                # Skip rows that don't have enough columns
                continue


def getprecautionDict():
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            precautionDictionary[row[0]] = row[1:]

def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    rf_clf = DecisionTreeClassifier().fit(X, y)
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1
    return rf_clf.predict([input_vector])

def calc_condition(exp, days):
    sum_severity = sum(severityDictionary[item] for item in exp)
    if (sum_severity * days) / (len(exp) + 1) > 13:
        return "You should consult a doctor."
    else:
        return "It might not be that bad, but you should take precautions."

# GUI Application
class HealthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Health Diagnosis System")
        self.geometry("800x600")
        self.configure(bg="#f0f8ff")

        self.symptom_vars = [tk.StringVar() for _ in range(5)]
        self.symptoms_list = sorted(symptoms_dict.keys())

        self.create_widgets()
        getSeverityDict()
        getDescription()
        getprecautionDict()

    def create_widgets(self):
        tk.Label(self, text="Health Diagnosis System", font=("Helvetica", 18, "bold"), bg="#f0f8ff").pack(pady=20)

        # Symptom selection
        for i in range(5):
            tk.Label(self, text=f"Symptom {i+1}:", bg="#f0f8ff").pack()
            symptom_cb = ttk.Combobox(self, textvariable=self.symptom_vars[i], values=self.symptoms_list)
            symptom_cb.pack()

        tk.Label(self, text="Number of Days:", bg="#f0f8ff").pack(pady=10)
        self.days_var = tk.IntVar()
        tk.Entry(self, textvariable=self.days_var).pack()

        tk.Button(self, text="Diagnose", command=self.diagnose, bg="#008080", fg="white").pack(pady=20)

    def diagnose(self):
        symptoms_entered = [var.get() for var in self.symptom_vars if var.get()]
        num_days = self.days_var.get()

        if not symptoms_entered or not num_days:
            messagebox.showerror("Input Error", "Please enter all the symptoms and number of days.")
            return

        # Prediction
        input_vector = np.zeros(len(symptoms_dict))
        for symptom in symptoms_entered:
            index = symptoms_dict.get(symptom)
            if index is not None:
                input_vector[index] = 1

        prediction = clf.predict([input_vector])
        disease = le.inverse_transform(prediction)[0]
        condition = calc_condition(symptoms_entered, num_days)

        # Display Results
        result_message = f"You may have: {disease}\n\nDescription:\n{description_list[disease]}\n\n{condition}\n\nPrecautions:"
        precautions = precautionDictionary[disease]
        for i, precaution in enumerate(precautions, 1):
            result_message += f"\n{i}. {precaution}"

        messagebox.showinfo("Diagnosis Result", result_message)

if __name__ == "__main__":
    app = HealthApp()
    app.mainloop()
