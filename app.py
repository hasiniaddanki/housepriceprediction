import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title='House Price Prediction', layout='centered')
st.title('House Price Prediction')

@st.cache_resource
def load_artifacts():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        enc = pickle.load(f)
    return model, enc

model, enc = load_artifacts()
le_location = enc['le_location']
le_transaction = enc['le_transaction']
le_furnishing = enc['le_furnishing']
le_status = enc['le_status']
features = enc['features']

st.markdown('Enter the house details and click Predict')

col1, col2 = st.columns(2)
with col1:
    bhk = st.selectbox('BHK', [1,2,3,4,5], index=2)
    carpet = st.number_input('Carpet Area (sqft)', min_value=100, max_value=10000, value=1200)
    bathrooms = st.selectbox('Bathrooms', [1,2,3,4,5], index=1)

with col2:
    location = st.selectbox('Location', list(le_location.classes_), index=0)
    transaction = st.selectbox('Transaction', list(le_transaction.classes_), index=0)
    furnishing = st.selectbox('Furnishing', list(le_furnishing.classes_), index=2)
    status = st.selectbox('Status', list(le_status.classes_), index=0)

if st.button('Predict'):
    def safe_transform(le, val):
        if val in list(le.classes_):
            return int(le.transform([val])[0])
        else:
            return int(0)

    loc_enc = safe_transform(le_location, location)
    trans_enc = safe_transform(le_transaction, transaction)
    furn_enc = safe_transform(le_furnishing, furnishing)
    stat_enc = safe_transform(le_status, status)

    x = np.array([[bhk, carpet, bathrooms, loc_enc, trans_enc, furn_enc, stat_enc]])
    pred_lakhs = model.predict(x)[0]
    pred_rupees = pred_lakhs * 100000

    st.success(f'Predicted price: ₹ {pred_rupees:,.0f} ({pred_lakhs:.2f} Lakhs)')
    st.caption('Random Forest model')

st.markdown('---')
st.markdown('Run `python train_and_save.py` to retrain the model.')
