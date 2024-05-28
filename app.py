from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('lgmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def calculate_egfr(features):
    # Extract the necessary features for eGFR calculation
    age, sc = features[0][:2]
    # Calculate eGFR using the CKD-EPI formula
    egfr = 144 * (min(sc / 0.7, 1) ** 0.7) * (max(sc / 0.7, 1) ** -0.329) * (0.993 ** age)
    return egfr

def CheckGfr(egfr):
    if egfr >= 90:
        stage = "Stage 1"
        cure = 1
    elif egfr >= 60:
        stage = "Stage 2"
        cure = 2
    elif egfr >= 30:
        stage = "Stage 3"
        cure = 3
    elif egfr >= 15:
        stage = "Stage 4"
        cure = 4
    elif egfr >= 0:
        stage = "Stage 5"
        cure = 5
    elif egfr < 0:
        stage = "GFR value cannot be negative"
    return stage, cure

@app.route('/')
def home():
    return render_template('front_page.html')

@app.route("/home")
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [[float(x) for x in request.form.values()]]
    # Calculate eGFR
    egfr = calculate_egfr(features)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    pred_proba = model.predict_proba(scaled_features)
    output_proba = '{0:.{1}f}'.format(pred_proba[0][1], 2)
    stage, cure = CheckGfr(egfr)
    if prediction[0] == 0.:
        output = 'DOES NOT HAVE CKD'
        has_stage = ""
        cure = 0
    else:
        output = 'HAS CKD'
        has_stage = " and is in {}".format(stage)

    return render_template('result.html', prediction_text='THE PATIENT {} with the probability {} {}'.format(output, output_proba, has_stage), cure=cure, stage=stage)

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/algorithm')
def algorithm():
    return render_template('algorithm.html')

if __name__ == "__main__":
    app.run(debug=False)
