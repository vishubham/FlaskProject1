import pickle
from flask import Flask, render_template, request
import numpy as np

# Create flask app
app = Flask(__name__)

# Load the pickeled model
rfcmodel = pickle.load(open('rfcmodel.pkl','rb'))

# Create flask endpoints
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_vals = [float(x) for x in request.form.values()]
    vals = [np.array(float_vals)]

    predictions = rfcmodel.predict(vals)
    
    return render_template('index.html', prediction=predictions)

if __name__ == '__main__':
    app.run(debug=True)
