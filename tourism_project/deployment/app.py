"""
app.py
Streamlit Web Application for Tourism Package Purchase Prediction.
Loads the trained model from tourism_project/deployment/ and provides
a user interface for making predictions.
"""

import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(model_path):
        # Try relative path from repo root (for Streamlit Cloud)
        model_path = "tourism_project/deployment/model.pkl"
    return joblib.load(model_path)

model = load_model()

# App title and description
st.title("✈️ Wellness Tourism Package Prediction")
st.markdown("""
This app predicts whether a customer will purchase the **Wellness Tourism Package**.  
Fill in the customer details below and click **Predict** to get the result.
""")

st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col2:
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    number_of_trips = st.number_input("Number of Trips (Annual)", min_value=1, max_value=20, value=2)
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income (₹)", min_value=5000, max_value=100000, value=20000)

st.divider()

# Predict button
if st.button("🔮 Predict", type="primary", use_container_width=True):
    # Create input dataframe matching training features
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": float(number_of_followups),
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": float(preferred_property_star),
        "MaritalStatus": marital_status,
        "NumberOfTrips": float(number_of_trips),
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": float(number_of_children_visiting),
        "Designation": designation,
        "MonthlyIncome": float(monthly_income)
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Display result
    st.markdown("### Prediction Result")
    if prediction == 1:
        st.success(f"✅ The customer is **likely to purchase** the Wellness Tourism Package! (Confidence: {probability[1]*100:.1f}%)")
    else:
        st.error(f"❌ The customer is **unlikely to purchase** the Wellness Tourism Package. (Confidence: {probability[0]*100:.1f}%)")

    # Show input summary
    with st.expander("View Input Summary"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}))
