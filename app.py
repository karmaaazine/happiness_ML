import streamlit as st
import joblib
import pandas as pd

# Load the model and preprocessor
model = joblib.load('happiness_predictor.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title('Happiness Predictor')

# Create input fields for key features
st.header('Enter your information')

# Example features - adjust based on your actual features
year = st.number_input('Year', min_value=1990, max_value=2023, value=1994)
workstat = st.selectbox('Work Status', ['working fulltime', 'working parttime', 'keeping house', 'retired', 'other'])
prestige = st.number_input('Prestige Score', min_value=0, max_value=100, value=50)
educ = st.number_input('Education Years', min_value=0, max_value=20, value=12)
income = st.selectbox(
    'Income',
    [
        '$10000 - 14999',
        '$15000 - 19999',
        '$20000 - 24999',
        '$25000 - 29999'
    ]
)
region = st.selectbox(
    'Region', 
    [
        'middle atlantic', 
        'new england', 
        'pacific', 
        'e. nor. central', 
        'south atlantic'
    ]
)
attend = st.selectbox('Religious Attendance', ['never', 'once a year', 'sevrl times a yr', 'every week'])

# Create a dictionary from inputs
input_data = {
    'year': [year],
    'workstat': [workstat],
    'prestige': [prestige],
    'educ': [educ],
    'income': [income],
    'region': [region]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

if st.button('Predict Happiness'):
    # Preprocess the input
    processed_input = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)
    
    # Display results
    if prediction[0] == 1:
        st.success('Prediction: Happy')
    else:
        st.warning('Prediction: Not Happy')
    
    st.write(f'Probability of being happy: {prediction_proba[0][1]:.2f}')