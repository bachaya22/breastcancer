from flask import Flask, request, render_template
import numpy as np
import pickle

# Load your trained model (replace 'model.pkl' with the correct path to your model)
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':




        input_text = request.form.get('input_data')

        try:
            # Convert the input text into a numpy array of floats
            input_data = np.array([float(i) for i in input_text.split(',')])

            # Reshape the input data to match the model's expected input shape
            input_data_reshaped = input_data.reshape(1, -1)

            # Make the prediction using the trained model
            prediction = model.predict(input_data_reshaped)

            # Determine the result based on the model's prediction
            if prediction[0] == 1:
                result = "Cancerous"
            else:
                result = "Not Cancerous"

            # Return the result to be displayed on the webpage
            return render_template('index.html', result=result)

        except ValueError:
            # Handle invalid input (non-numeric or missing values)
            return render_template('index.html', result="Invalid input. Please provide comma-separated numeric values.")

if __name__ == '__main__':
    app.run(debug=True)
