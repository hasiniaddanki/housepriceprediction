import streamlit as st

st.set_page_config(page_title='House Price Prediction', layout='centered')
st.title('House Price Prediction')
st.write('App is loading...')

try:
    import numpy as np
    st.success('NumPy loaded')
except Exception as e:
    st.error(f'NumPy error: {e}')

try:
    import joblib
    st.success('Joblib loaded')
except Exception as e:
    st.error(f'Joblib error: {e}')

try:
    import sklearn
    st.success(f'Scikit-learn loaded: {sklearn.__version__}')
except Exception as e:
    st.error(f'Scikit-learn error: {e}')

import os
st.write('---')
st.write('**Files in directory:**')
for f in os.listdir('.'):
    st.write(f'- {f}')

st.write('---')

if os.path.exists('rf_model.joblib') and os.path.exists('encoders.joblib'):
    st.success('Model files found!')
    try:
        model = joblib.load('rf_model.joblib')
        enc = joblib.load('encoders.joblib')
        st.success('Model loaded successfully!')
        
        le_location = enc['le_location']
        le_transaction = enc['le_transaction']
        le_furnishing = enc['le_furnishing']
        le_status = enc['le_status']
        
        st.write('---')
        st.subheader('Make a Prediction')
        
        col1, col2 = st.columns(2)
        with col1:
            bhk = st.selectbox('BHK', [1,2,3,4,5], index=2)
            carpet = st.number_input('Carpet Area (sqft)', min_value=100, max_value=10000, value=1200)
            bathrooms = st.selectbox('Bathrooms', [1,2,3,4,5], index=1)

        with col2:
            location = st.selectbox('Location', list(le_location.classes_)[:20], index=0)
            transaction = st.selectbox('Transaction', list(le_transaction.classes_), index=0)
            furnishing = st.selectbox('Furnishing', list(le_furnishing.classes_), index=0)
            status = st.selectbox('Status', list(le_status.classes_), index=0)

        if st.button('Predict Price'):
            loc_enc = le_location.transform([location])[0] if location in le_location.classes_ else 0
            trans_enc = le_transaction.transform([transaction])[0]
            furn_enc = le_furnishing.transform([furnishing])[0]
            stat_enc = le_status.transform([status])[0]
            
            import numpy as np
            x = np.array([[bhk, carpet, bathrooms, loc_enc, trans_enc, furn_enc, stat_enc]])
            pred = model.predict(x)[0]
            
            st.success(f'Predicted Price: Rs {pred*100000:,.0f} ({pred:.2f} Lakhs)')
    except Exception as e:
        st.error(f'Error loading model: {e}')
        import traceback
        st.code(traceback.format_exc())
else:
    st.error('Model files not found!')
    st.write('Looking for: rf_model.joblib, encoders.joblib')
