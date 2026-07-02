import streamlit as st
import pandas as pd
import pickle
import xgboost

@st.cache_resource
def load_model():
    loaded_model = pickle.load(open('model_and_mapping.pkl', 'rb'))
    return loaded_model["model"], loaded_model["location_mapping"]

model, location_mapping = load_model()



st.set_page_config(page_title=" House Price Predictor", layout="centered", page_icon="🏠")

# Custom CSS for Premium Design
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.02);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    div.stNumberInput > label {
        font-weight: bold;
        color: #495057;
    }
    .main-title {
        color: #1e293b;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown("<h1 class='main-title'>California House Price Predictor</h1>", unsafe_allow_html=True)

# Main container for inputs
with st.container():
    st.markdown("### 📍 Property Details")
    
    # Location section
    location_names = list(location_mapping.keys())
    selected_location = st.selectbox("Select Location", location_names, index=0)
    
    # Get location data
    loc_data = location_mapping[selected_location]
    population = loc_data["Population"]
    latitude = loc_data["Latitude"]
    longitude = loc_data["Longitude"]

    st.info(f"📍 Location Stats: Latitude {latitude}, Longitude {longitude}, Population {population:,.0f}")

    st.divider()
    st.markdown("### 🏗️ House Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bedrooms = st.number_input("🏨 Number of Bedrooms", min_value=1, max_value=10, value=3)
        kitchens = st.number_input("🍳 Number of Kitchens", min_value=1, max_value=5, value=1)
        
    with col2:
        bathrooms = st.number_input("🚿 Number of Bathrooms", min_value=1, max_value=10, value=2)
        house_age = st.number_input("📅 House Age (Years)", min_value=0, max_value=100, value=15)

# Prediction Logic
st.markdown("---")
if st.button("✨ Predict Property Value"):
    with st.spinner("Analyzing market data..."):
        # Mapping frontend inputs to model features
        # Assuming AveRooms = bedrooms + kitchens + bathrooms + 1 (living area)
        avg_rooms = bedrooms + kitchens + bathrooms + 1
        avg_bedrooms = bedrooms
        # Using the dataset mean for Median Income as it's not requested but critical for the model
        med_inc = 3.87 

        input_df = pd.DataFrame([{
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": avg_rooms,
            "AveBedrms": avg_bedrooms,
            "Population": population,
            "Latitude": latitude,
            "Longitude": longitude,
        }])

        prediction = model.predict(input_df)[0]
        
        price_usd = prediction * 100000
        price_inr = price_usd * 85  

        st.markdown("### 💰 Prediction Results")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.success(f"**USD Valuation**\n\n### ${price_usd:,.2f}")
            
        with res_col2:
            st.success(f"**INR Valuation**\n\n### ₹{price_inr:,.2f}")

        st.balloons()

   
    
