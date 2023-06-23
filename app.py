from flask import Flask

import pickle
from flask import request, jsonify

app = Flask(__name__)

gender_map = {"F": 0, "M": 1}
lung_cancer_map = {0: "NO", 1: "YES"}

def predict_lung_cancer(GENDER,
                        AGE,
                        SMOKING,
                        YELLOW_FINGERS,
                        ANXIETY,
                        PEER_PRESSURE,
                        CHRONIC_DISEASE,
                        FATIGUE,
                        ALLERGY,
                        WHEEZING,
                        ALCOHOL_CONSUMING,
                        COUGHING,
                        SHORTNESS_OF_BREATH,
                        SWALLOWING_DIFFICULTY,
                        CHEST_PAIN):
    # 1. Read the machine learning model from its saved state ...
    pickle_file = open('model.pkl', 'rb')
    model = pickle.load(pickle_file)

    # 2. Transform the "raw data" passed into the function to the encoded / numerical values using the maps / dictionaries
    GENDER = gender_map[GENDER]

    # 3. Make an individual prediction for this set of data
    y_predict = model.predict([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE,
                                CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING,
                                COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]])[0]

    # 4. Return the "raw" version of the prediction i.e. the actual name of the drug rather than the numerical encoded version
    return lung_cancer_map[y_predict]

@app.route("/")
def hello():
    return "A test web service for accessing a machine learning model to make Lung Cancer Prediction."

@app.route('/drug', methods=['GET'])
def api_all():
#    return jsonify(data_science_books)

    GENDER = request.args['GENDER']
    AGE = int(request.args['AGE'])
    SMOKING = int(request.args['SMOKING'])
    YELLOW_FINGERS = int(request.args['YELLOW_FINGERS'])
    ANXIETY = int(request.args['ANXIETY'])
    PEER_PRESSURE = int(request.args['PEER_PRESSURE'])
    CHRONIC_DISEASE = int(request.args['CHRONIC_DISEASE'])
    FATIGUE = int(request.args['FATIGUE'])
    ALLERGY = int(request.args['ALLERGY'])
    WHEEZING = int(request.args['WHEEZING'])
    ALCOHOL_CONSUMING = int(request.args['ALCOHOL_CONSUMING'])
    COUGHING = int(request.args['COUGHING'])
    SHORTNESS_OF_BREATH = int(request.args['SHORTNESS_OF_BREATH'])
    SWALLOWING_DIFFICULTY = int(request.args['SWALLOWING_DIFFICULTY'])
    CHEST_PAIN = int(request.args['CHEST_PAIN'])


    Lung_Cancer = predict_lung_cancer(GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, 
                                      CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING,
                                      COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN)

    #return(jsonify(drug))
    return(jsonify(predict_lung_cancer = Lung_Cancer))