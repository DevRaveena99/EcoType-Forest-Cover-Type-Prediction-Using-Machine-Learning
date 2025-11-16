
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='EcoType - Forest Cover Prediction')
st.title('EcoType: Forest Cover Type Prediction')

# Load artifacts
MODEL_PATH = 'C:/Users/user/Downloads/Forest_Cover_Classification/best_model.pkl'
SCALER_PATH = 'C:/Users/user/Downloads/Forest_Cover_Classification/scaler.pkl'
TARGET_ENCODER_PATH = 'C:/Users/user/Downloads/Forest_Cover_Classification/target_encoder.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(TARGET_ENCODER_PATH)

# Column Groups
numeric_cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
wilderness_cols = [c for c in model.feature_names_in_ if c.startswith('Wilderness_Area')]
soil_cols = [c for c in model.feature_names_in_ if c.startswith('Soil_Type')]

# Input fields
st.sidebar.header('Input features')
input_data = {}

for col in numeric_cols:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

for col in wilderness_cols + soil_cols:
    input_data[col] = st.sidebar.selectbox(col, options=[0,1], index=0)

input_df = pd.DataFrame([input_data])

# ---------------- FEATURE ENGINEERING ----------------
input_df['Hillshade_diff_9noon'] = input_df['Hillshade_9am'] - input_df['Hillshade_Noon']
input_df['Hillshade_diff_noon3'] = input_df['Hillshade_Noon'] - input_df['Hillshade_3pm']
input_df['Hydro_dist_ratio'] = input_df['Vertical_Distance_To_Hydrology'] / (input_df['Horizontal_Distance_To_Hydrology'] + 1)
input_df['Slope_Elev_ratio'] = input_df['Slope'] / (input_df['Elevation'] + 1)
input_df['Total_Road_Hydro'] = input_df['Horizontal_Distance_To_Roadways'] + input_df['Horizontal_Distance_To_Hydrology']
input_df['Soil_Count'] = input_df[soil_cols].sum(axis=1)
input_df['Wilderness_Count'] = input_df[wilderness_cols].sum(axis=1)

# Scale numeric
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Prediction
if st.button('Predict'):
    pred_enc = model.predict(input_df)
    pred_label = le.inverse_transform(pred_enc)
    st.success('Predicted Cover Type: {}'.format(pred_label[0]))
