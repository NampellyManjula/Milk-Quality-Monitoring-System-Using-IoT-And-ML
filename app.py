from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read the data
    milkData = pd.read_csv(r'C:\\Users\\Lenovo\\Downloads\\milknew.csv')

    # Preprocessing steps

    # Drop rows with missing values
    milkData.dropna(inplace=True)

    # Convert the 'Grade' column to numerical values
    milkData['Grade'].replace({'high': 2, 'medium': 1, 'low': 0}, inplace=True)

    # Split the data into input features (X) and target variable (y)
    X = milkData.drop(['Grade'], axis=1)
    y = milkData['Grade']

    # Train the model
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    model.fit(X, y)

    # Retrieve user input values
    pH = request.form['pH']
    temp = request.form['temp']
    taste = request.form['taste']
    odor = request.form['odor']
    fat = request.form['fat']
    turbidity = request.form['turbidity']
    color = request.form['color']

    # Handle None values and convert input values to float
    try:
        pH = float(pH)
        temp = float(temp)
        taste = float(taste)
        odor = float(odor)
        fat = float(fat)
        turbidity = float(turbidity)
        color = float(color)
    except ValueError:
        # Handle the error case where the input values cannot be converted to float
        return "Invalid input values"

    # Make predictions for the user input
    prediction = model.predict([[pH, temp, taste, odor, fat, turbidity, color]])

    # Convert prediction to human-readable format
    if prediction == 0:
        quality = 'Low Quality'
    elif prediction == 1:
        quality = 'Medium Quality'
    else:
        quality = 'High Quality'

    return render_template('index.html', prediction=quality)

if __name__ == '__main__':
    app.run(debug=True)