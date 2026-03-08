import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Used Bike Price Predictor", page_icon="🏍️")

st.title("🏍️ Used Bike Price Predictor")
st.write("Apni bike ki details enter karein aur market price ka andaza lagayein.")

# --- LOAD MODEL ---
@st.cache_resource
def load_bike_model():
    with open('bike_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    data = load_bike_model()
    model = data['model']
    le_dict = data['le_dict']
    features = data['features']
    dropdown_options = data['dropdown_options']
except:
    st.error("Pehle 'train_bike.py' run karein taaki model file ban sake.")
    st.stop()

# --- USER INPUT ---
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Bike Brand", dropdown_options['brand'])
    owner = st.selectbox("Owner Type", dropdown_options['owner'])
    power = st.number_input("Engine Power (cc)", min_value=50, max_value=2000, value=150)

with col2:
    age = st.number_input("Bike Age (Years)", min_value=0, max_value=50, value=3)
    kms = st.number_input("Kilometers Driven", min_value=0, value=10000)

# --- PREDICTION ---
if st.button("Predict Bike Price", use_container_width=True):
    # Encoding inputs
    brand_enc = le_dict['brand'][1].index(brand)
    owner_enc = le_dict['owner'][1].index(owner)
    
    
    input_data = [[brand_enc, owner_enc, kms, age, power]]
    
    # Prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f"### Estimated Price: ₹{round(prediction, 2)}")
    st.info("Note: Ye price market trends aur aapke data par based hai.")