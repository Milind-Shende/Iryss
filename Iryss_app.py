import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np

ROOT_DIR = os.getcwd()
SAVED_DIR_PATH = "saved_models"
SAVED_ZERO_FILE="0"
MODEL_FILE_DIR ="model"
MODEL_FILE_NAME = "model.pkl"
TRANSFORMER_FILE_DIR="transformer"
TRANSFORMER_FILE_NAME="transformer.pkl"
# TARGET_ENCODER_FILE_DIR="target_encoder"
# TARGET_ENCODER_FILE_NAME="target_encoder.pkl"

MODEL_DIR = os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,MODEL_FILE_DIR,MODEL_FILE_NAME)
# print("MODEL_PATH:-",MODEL_DIR)

TRANSFORMER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TRANSFORMER_FILE_DIR,TRANSFORMER_FILE_NAME)
# print("TRANSFORMER_PATH:-",TRANSFORMER_DIR)

# TARGET_ENCODER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TARGET_ENCODER_FILE_DIR,TARGET_ENCODER_FILE_NAME)
# print("TARGET_ENCODER_PATH:-",TARGET_ENCODER_DIR)

# Load the Model.pkl, Transformer.pkl and Target.pkl
model=pickle.load(open(MODEL_DIR,"rb"))
# print(model)
transfomer=pickle.load(open(TRANSFORMER_DIR,"rb"))
# print(transfomer)



# About page
def about_page():
    st.title('Predicting the Financial Burden of Lung Cancer')
    st.write('The project aims to develop a predictive model that estimates the annual out-of-pocket costs for patients diagnosed with Stage 3&4 lung cancer. By considering factors such as age, comorbidities, and primary insurance, the model will enable patients to proactively plan for future financial burdens associated with their diagnosis. The ultimate goal is to alleviate the financial stress and reduce the likelihood of personal bankruptcy that over 40% of cancer patients experience within four years of diagnosis.')
    
def visualization_page():...


# Main prediction page
def prediction_page():
    # Title and input fields
    st.title('Predicting the Financial Burden of Lung Cancer')
    st.subheader('Patient Information')
    AGE = st.number_input('Age', min_value=0, max_value=120, value=30)
    SEX = st.selectbox('Gender', ('Male', 'Female', 'Other'))
    RACE = st.selectbox('Race', ('Black', 'Other', 'White', 'Hispanic', 'Native American', 'Asian or Pacific Islander'))
    HOSPID = st.text_input('Hospital ID')
    NCHRONIC = st.selectbox('Numbder of Chronic Diseases', ('1', '2'))
    ZIPINC_QRTL = st.selectbox('Income Level to ZIP code', ('1', '2', '3'))
    NPR = st.number_input('Net Patient Revenue')
    # Add input fields for other features as needed
    
    # Hospital and Insurance Information
    st.subheader('Hospital and Insurance Information')
    DRG= st.selectbox('Diagnosis Related Group', ('ICD-10-CM', 'ICD-10-CM/PCS', 'ICD-9-CM', 'ICD-10-PCS'))
    DXn = st.selectbox('Level of disease diagnosis', ('3', '4'))
    CM_DRUG = st.selectbox('Drug Intake', ('current', 'never', 'former'))
    PAY1 = st.selectbox('Payment Method', ('Medicare', 'Medicaid', 'Private including HMO', 'Self-Pay', 'No charge', 'Other'))
    PAY2 = st.selectbox('Insurance Company', ('COBRA Coverage', 'Secondary Health Insurance', 'Employer-Sponsored Plans', 'Government Programs', 'NONE'))

    # Checkbox for presence of medical conditions
    st.subheader('Medical Conditions')
    CM_AIDS = st.checkbox('AIDS')
    CM_ALCOHOL = st.checkbox('Alcohol Consumption')
    CM_ANEMDEF = st.checkbox('Congenital Monosomy with Anemia and Defects')
    CM_ARTH = st.checkbox('Arthritis')
    CM_BLDLOSS = st.checkbox('Blood Loss')
    CM_CHF = st.checkbox('Congestive Heart Failure')
    TRAN_IN= st.selectbox('Transfer patient In', ('Transferred from acute care hospital', 'Not a transfer', 'Transferred from another health facility', 'Transferred from '))
    TRAN_OUT = st.selectbox('Transfer patient Out', ('Not a transfer', 'Transferred out to acute care hospital', 'Transferred out to another health facility'))
    
     
    # Prediction button
    if st.button('Predict'):
        # Preprocess the input features
        input_data = {
            'AGE': [AGE],
            'SEX': [SEX],
            'RACE': [RACE],
            'HOSPID':[HOSPID],
            'NCHRONIC':[NCHRONIC],
            'ZIPINC_QRTL':[ZIPINC_QRTL],
            'NPR':[NPR],
            'DRG':[DRG],
            'DXn':[DXn],
            'CM_DRUG':[CM_DRUG],
            'PAY1':[PAY1],
            'PAY2':[PAY2],
            'CM_AIDS': ['yes' if CM_AIDS else 'no'],
            'CM_ALCOHOL': ['yes' if CM_ALCOHOL else 'no'],
            'CM_ANEMDEF': ['yes' if CM_ANEMDEF else 'no'],
            'CM_ARTH': ['yes' if CM_ARTH else 'no'],
            'CM_BLDLOSS': ['yes' if CM_BLDLOSS else 'no'],
            'CM_CHF': ['yes' if CM_CHF else 'no'],
            'TRAN_IN':[TRAN_IN],
            'TRAN_OUT':[TRAN_OUT]     
        }
        # Convert input data to a Pandas DataFrame
        input_df = pd.DataFrame(input_data)
        # Perform the transformation using the loaded transformer
        transformed_data = transfomer.transform(input_df)
        # Reshape the transformed data as a NumPy array
        input_arr = np.array(transformed_data)
        

        # Make the prediction using the loaded model
        prediction = model.predict(input_arr)
        st.subheader('Prediction')
        st.write(f'The predicted total charge is: {prediction[0]:.2f}')

# Teams page
def collaborators_page():
    st.title('Predicting the Financial Burden of Lung Cancer')
    st.write('Meet our awesome team members:')
    st.write('- Team Member 1')
    st.write('- Team Member 2')
    st.write('- Team Member 3')
    # Add more team members as needed

# Create a dictionary with page names and their corresponding functions
pages = {
    'About': about_page,
    'Visualization ':visualization_page, 
    'Prediction': prediction_page,
    'Collaborators': collaborators_page,
}

# Streamlit application
def main():
    # Sidebar navigation
    st.sidebar.title('Navigation')
    selected_page = st.sidebar.radio('Go to', list(pages.keys()))

    # Display the selected page content
    pages[selected_page]()

# Run the Streamlit application
if __name__ == '__main__':
    main()