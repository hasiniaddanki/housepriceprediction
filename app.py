import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title='House Price Prediction', layout='centered')

# load model and encoders
model = joblib.load('rf_model.joblib')
enc = joblib.load('encoders.joblib')

le_location = enc['le_location']
le_transaction = enc['le_transaction']
le_furnishing = enc['le_furnishing']
le_status = enc['le_status']

# app title
st.title('🏠 House Price Prediction')
st.write('Enter the details below to predict house price')

st.divider()

# input form
col1, col2 = st.columns(2)

with col1:
    bhk = st.selectbox('BHK', [1, 2, 3, 4, 5], index=2)
    carpet = st.number_input('Carpet Area (sqft)', min_value=100, max_value=10000, value=1200)
    bathrooms = st.selectbox('Bathrooms', [1, 2, 3, 4, 5], index=1)

with col2:
    location = st.selectbox('Location', list(le_location.classes_)[:20])
    transaction = st.selectbox('Transaction', list(le_transaction.classes_))
    furnishing = st.selectbox('Furnishing', list(le_furnishing.classes_))
    status = st.selectbox('Status', list(le_status.classes_))

st.divider()

# predict button
if st.button('Predict Price', type='primary'):
    # encode inputs
    loc_enc = le_location.transform([location])[0] if location in le_location.classes_ else 0
    trans_enc = le_transaction.transform([transaction])[0]
    furn_enc = le_furnishing.transform([furnishing])[0]
    stat_enc = le_status.transform([status])[0]
    
    # make prediction
    x = np.array([[bhk, carpet, bathrooms, loc_enc, trans_enc, furn_enc, stat_enc]])
    pred = model.predict(x)[0]
    
    # show result
    st.success(f'### Predicted Price: ₹{pred*100000:,.0f}')
    st.caption(f'({pred:.2f} Lakhs)')

# footer
st.divider()
st.caption('Built with Streamlit | ML Model: Random Forest')
