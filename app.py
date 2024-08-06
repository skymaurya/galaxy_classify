#make sure to save  a folder with name templates
# UI: How many inputs you have (You will be taking from user), dropdown , text
# Be ready model.pkl
# 3 steps: 1. Take input from USer
#          2. Pass that input to the model
#          3. Take the result received from model and display it

from flask import Flask, render_template, request
import pickle
import pandas as pd

# Create a Flask application instance
app = Flask(__name__)

# Load the trained model
with open('clf.pkl', 'rb') as file:
    model = pickle.load(file)

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle form submission
@app.route("/submit", methods=["POST"])
def submit():
    # Reading input values from the form
    input_feature = [float(x) for x in request.form.values()]
    names = ['r', 'i', 'z', 'petroRad_g', 'petroRad_r', 'petroR50_u', 'petroR50_g', 'petroR50_i', 'petroR50_r', 'petroR50_z']
    
    # Create DataFrame from input features
    df = pd.DataFrame([input_feature], columns=names)

    # Make prediction
    prediction = model.predict(df)
    
    # Render the output template with the predictions result
    if prediction == 0:
        print(prediction)
        return render_template("inner-page.html",prediction="Starforming")
    else:
        print(prediction)
        return render_template("inner-page.html",prediction="Starbursting")
# Run the application
if __name__ == "__main__":
    app.run(debug=True,port=5000)