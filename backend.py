import sys
import json
import numpy as np
import joblib
import pandas as pd

# Load machine learning models
knn_model = joblib.load('knn_model.joblib')
lr_model = joblib.load('lr_model.joblib')
rf_model = joblib.load('rf_model.joblib')
stacked_ensemble_model = joblib.load('stacked_ensemble_model.joblib')
weighted_ensemble_model = joblib.load('weighted_ensemble_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')
#print(knn_model.feature_names_in_)
def preprocess_input(input_data):
    # Map chest_pain values
    chest_pain_mapping = {
        0: 'asymptomatic',
        1: 'atypical_angina',
        2: 'non-anginal_pain',
        3: 'typical_angina'
    }
    input_data['chest_pain'] = chest_pain_mapping.get(input_data['cp'], input_data['cp'])

    # Map rest_ecg values
    rest_ecg_mapping = {
        0: 'left_ventricular_hypertrophy',
        1: 'normal',
        2: 'STT_abnormality'
    }
    input_data['rest_ecg'] = rest_ecg_mapping.get(input_data['restecg'], input_data['restecg'])

    # Map st_slope_type values
    st_slope_mapping = {
        0: 'downsloping',
        1: 'flat',
        2: 'upsloping'
    }
    input_data['st_slope_type'] = st_slope_mapping.get(input_data['slope'], input_data['slope'])

    # Map thalassemia values
    thalassemia_mapping = {
        0: 'nothing',
        1: 'fixed',
        2: 'normal',
        3: 'reversible'
    }
    input_data['thalassemia'] = thalassemia_mapping.get(input_data['thal'], input_data['thal'])

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])


    # Set values for each of the expected columns based on user input
    input_df['age'] = input_data['age']
    input_df['sex'] = input_data['sex']
    input_df['resting_blood_pressure'] = input_data['trestbps']
    input_df['cholesterol'] = input_data['chol']
    input_df['fasting_blood_sugar'] = 1 if input_data['fbs'] > 120 else 0
    input_df['max_heart_rate'] = input_data['thalach']
    input_df['exercise_induced_angina'] = input_data['exang']
    input_df['st_depression'] = input_data['oldpeak']
    input_df['num_major_vessels'] = input_data['ca']

    # Set chest pain columns
    input_df[['chest_pain_asymptomatic', 'chest_pain_atypical_angina', 'chest_pain_non-anginal_pain', 'chest_pain_typical_angina']] = 0
    chest_pain_col = f'chest_pain_{input_data["chest_pain"]}'
    input_df[chest_pain_col] = 1

    # Set rest_ecg columns
    input_df[['rest_ecg_STT_abnormality', 'rest_ecg_left_ventricular_hypertrophy', 'rest_ecg_normal']] = 0
    rest_ecg_col = f'rest_ecg_{input_data["rest_ecg"]}'
    input_df[rest_ecg_col] = 1

    # Set st slope type columns
    input_df[['st_slope_type_downsloping', 'st_slope_type_flat', 'st_slope_type_upsloping']] = 0
    st_slope_col = f'st_slope_type_{input_data["st_slope_type"]}'
    input_df[st_slope_col] = 1

    # Set thalassemia columns
    input_df[['thalassemia_fixed', 'thalassemia_normal', 'thalassemia_reversible']] = 0
    thal_col = f'thalassemia_{input_data["thalassemia"]}'
    input_df[thal_col] = 1

    # Reorder columns to match the expected order
    expected_columns = [
        'age', 'sex', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'max_heart_rate', 'exercise_induced_angina', 'st_depression',
        'num_major_vessels', 'chest_pain_asymptomatic', 'chest_pain_atypical_angina', 'chest_pain_non-anginal_pain',
        'chest_pain_typical_angina', 'rest_ecg_STT_abnormality', 'rest_ecg_left_ventricular_hypertrophy',
        'rest_ecg_normal', 'st_slope_type_downsloping', 'st_slope_type_flat', 'st_slope_type_upsloping',
        'thalassemia_fixed', 'thalassemia_normal', 'thalassemia_reversible'
    ]

    input_df = input_df[expected_columns]


    return input_df



# Read input data from Node.js server
input_data_str = sys.argv[1]

# Parse the JSON string into a dictionary
input_data = json.loads(input_data_str)

# Preprocess input data
preprocessed_input = preprocess_input(input_data)

preprocessed_input.to_csv('preprocessed_input.csv', index=False)
# Make predictions
knn_prediction = int(knn_model.predict(preprocessed_input)[0])
lr_prediction = int(lr_model.predict(preprocessed_input)[0])
rf_prediction = int(rf_model.predict(preprocessed_input)[0])
stacked_ensemble_prediction = int(stacked_ensemble_model.predict(preprocessed_input)[0])
weighted_ensemble_prediction = int(weighted_ensemble_model.predict(preprocessed_input)[0])
xgb_prediction = int(xgb_model.predict(preprocessed_input)[0])

# Return predictions as a JSON string
predictions = json.dumps({
    'knn_prediction': knn_prediction,
    'lr_prediction': lr_prediction,
    'rf_prediction': rf_prediction,
    'stacked_ensemble_prediction': stacked_ensemble_prediction,
    'weighted_ensemble_prediction': weighted_ensemble_prediction,
    'xgb_prediction': xgb_prediction
})

# Print predictions to stdout (this will be captured by Node.js)
print(predictions, flush=True)
