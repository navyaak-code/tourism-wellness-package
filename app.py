
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Tourism Wellness Package Prediction",
    page_icon="🏖️",
    layout="wide"
)

st.title("🏖️ Tourism Wellness Package Prediction")
st.write(
    "Enter the customer details below to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)


# --------------------------------------------------
# Load model from Hugging Face
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="navyaak-code/tourism-wellness-package-model",
        filename="best_tourism_model.pkl"
    )

    model = joblib.load(model_path)

    return model


model = load_model()


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    CityTier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    DurationOfPitch = st.number_input(
        "Duration of Pitch",
        min_value=0,
        value=10
    )

    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        value=3
    )

    NumberOfFollowups = st.number_input(
        "Number of Follow-ups",
        min_value=0,
        value=3
    )

    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    NumberOfTrips = st.number_input(
        "Number of Trips",
        min_value=0,
        value=3
    )

    Passport = st.selectbox(
        "Passport",
        [0, 1]
    )

    PitchSatisfactionScore = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

with col2:

    OwnCar = st.selectbox(
        "Own Car",
        [0, 1]
    )

    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        value=1
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=20000.0
    )

    TypeofContact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    Occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Free Lancer"
        ]
    )

    Gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    ProductPitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Deluxe",
            "Standard",
            "Super Deluxe",
            "King"
        ]
    )

with col3:

    MaritalStatus = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Unmarried",
            "Divorced"
        ]
    )

    Designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )


# --------------------------------------------------
# Create input dataframe
# --------------------------------------------------

if st.button("🔮 Predict"):

    input_data = pd.DataFrame({

        "Age": [Age],

        "CityTier": [CityTier],

        "DurationOfPitch": [DurationOfPitch],

        "NumberOfPersonVisiting": [NumberOfPersonVisiting],

        "NumberOfFollowups": [NumberOfFollowups],

        "PreferredPropertyStar": [PreferredPropertyStar],

        "NumberOfTrips": [NumberOfTrips],

        "Passport": [Passport],

        "PitchSatisfactionScore": [PitchSatisfactionScore],

        "OwnCar": [OwnCar],

        "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],

        "MonthlyIncome": [MonthlyIncome],

        "TypeofContact_Self Enquiry":
            [1 if TypeofContact == "Self Enquiry" else 0],

        "Occupation_Large Business":
            [1 if Occupation == "Large Business" else 0],

        "Occupation_Salaried":
            [1 if Occupation == "Salaried" else 0],

        "Occupation_Small Business":
            [1 if Occupation == "Small Business" else 0],

        "Gender_Male":
            [1 if Gender == "Male" else 0],

        "ProductPitched_Deluxe":
            [1 if ProductPitched == "Deluxe" else 0],

        "ProductPitched_King":
            [1 if ProductPitched == "King" else 0],

        "ProductPitched_Standard":
            [1 if ProductPitched == "Standard" else 0],

        "ProductPitched_Super Deluxe":
            [1 if ProductPitched == "Super Deluxe" else 0],

        "MaritalStatus_Married":
            [1 if MaritalStatus == "Married" else 0],

        "MaritalStatus_Single":
            [1 if MaritalStatus == "Single" else 0],

        "MaritalStatus_Unmarried":
            [1 if MaritalStatus == "Unmarried" else 0],

        "Designation_Executive":
            [1 if Designation == "Executive" else 0],

        "Designation_Manager":
            [1 if Designation == "Manager" else 0],

        "Designation_Senior Manager":
            [1 if Designation == "Senior Manager" else 0],

        "Designation_VP":
            [1 if Designation == "VP" else 0]
    })


    # Ensure exact feature order
    input_data = input_data[
        [
            "Age",
            "CityTier",
            "DurationOfPitch",
            "NumberOfPersonVisiting",
            "NumberOfFollowups",
            "PreferredPropertyStar",
            "NumberOfTrips",
            "Passport",
            "PitchSatisfactionScore",
            "OwnCar",
            "NumberOfChildrenVisiting",
            "MonthlyIncome",
            "TypeofContact_Self Enquiry",
            "Occupation_Large Business",
            "Occupation_Salaried",
            "Occupation_Small Business",
            "Gender_Male",
            "ProductPitched_Deluxe",
            "ProductPitched_King",
            "ProductPitched_Standard",
            "ProductPitched_Super Deluxe",
            "MaritalStatus_Married",
            "MaritalStatus_Single",
            "MaritalStatus_Unmarried",
            "Designation_Executive",
            "Designation_Manager",
            "Designation_Senior Manager",
            "Designation_VP"
        ]
    ]


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            f"✅ The customer is likely to purchase the Wellness Package."
        )

    else:

        st.warning(
            f"❌ The customer is unlikely to purchase the Wellness Package."
        )

    st.write(
        f"**Probability of purchase:** {probability:.2%}"
    )
