import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title='House Price Prediction',
    page_icon='🏠',
    layout='centered'
)

# custom css for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.1rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
        border: none;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .price-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .price-box h2 {
        color: white;
        font-size: 2rem;
        margin: 0;
    }
    .price-box p {
        color: #f0f0f0;
        font-size: 1rem;
    }
    .input-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load('rf_model.joblib')
    enc = joblib.load('encoders.joblib')
    return model, enc

model, enc = load_model()

le_location = enc['le_location']
le_transaction = enc['le_transaction']
le_furnishing = enc['le_furnishing']
le_status = enc['le_status']

# header
st.markdown("""
<div class="main-header">
    <h1>🏠 House Price Prediction</h1>
    <p>Enter property details to get an instant price estimate</p>
</div>
""", unsafe_allow_html=True)

# input section
st.subheader('📝 Property Details')

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🛏️ Rooms & Area")
    bhk = st.selectbox('Number of BHK', [1, 2, 3, 4, 5], index=2)
    carpet = st.number_input('Carpet Area (sqft)', min_value=100, max_value=10000, value=1200, step=50)
    bathrooms = st.selectbox('Number of Bathrooms', [1, 2, 3, 4, 5], index=1)

with col2:
    st.markdown("##### 📍 Location & Features")
    location = st.selectbox('Select Location', list(le_location.classes_)[:20])
    transaction = st.selectbox('Transaction Type', list(le_transaction.classes_))
    furnishing = st.selectbox('Furnishing Status', list(le_furnishing.classes_))
    status = st.selectbox('Property Status', list(le_status.classes_))

st.markdown("---")

# predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button('🔮 Predict Price', type='primary', use_container_width=True)

if predict_btn:
    # encode inputs
    loc_enc = le_location.transform([location])[0] if location in le_location.classes_ else 0
    trans_enc = le_transaction.transform([transaction])[0]
    furn_enc = le_furnishing.transform([furnishing])[0]
    stat_enc = le_status.transform([status])[0]
    
    # make prediction
    x = np.array([[bhk, carpet, bathrooms, loc_enc, trans_enc, furn_enc, stat_enc]])
    pred = model.predict(x)[0]
    price_inr = pred * 100000
    
    # show result with animation
    st.balloons()
    
    st.markdown(f"""
    <div class="price-box">
        <h2>💰 ₹{price_inr:,.0f}</h2>
        <p>Estimated Price: {pred:.2f} Lakhs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # show summary
    st.subheader('📊 Prediction Summary')
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("BHK", f"{bhk} BHK")
    with summary_col2:
        st.metric("Area", f"{carpet} sqft")
    with summary_col3:
        st.metric("Location", location.title())

# sidebar with info
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/home.png", width=150)
    st.title("About")
    st.info("""
    This app predicts house prices using a **Random Forest** machine learning model.
    
    **Features used:**
    - BHK (Bedrooms)
    - Carpet Area
    - Bathrooms
    - Location
    - Transaction Type
    - Furnishing
    - Status
    """)
    
    st.markdown("---")
    st.subheader("📈 Model Info")
    st.write("**Algorithm:** Random Forest")
    st.write("**Accuracy:** ~84% R² Score")
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit")

# footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🎓 House Price Prediction Project | Built with Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
