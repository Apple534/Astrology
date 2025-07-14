import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Astrology Predictor", layout="centered")

# 🔧 Custom CSS for black background and white text
st.markdown("""
    <style>
        body, .stApp {
            background-color: #000000;
            color: white;
        }
        /* Label color fix */
        label, .stSelectbox label, .stTextInput label, .stDateInput label {
            color: white !important;
            font-weight: 600;
        }

        /* Widget background */
        .css-1cpxqw2, .css-ffhzg2, .css-1d391kg, .css-1v3fvcr {
            background-color: #111111 !important;
            color: white !important;
        }

        .stSelectbox > div, .stTextInput > div {
            background-color: #111111 !important;
            color: white !important;
        }

        .stButton>button {
            background-color: #333;
            color: white;
            border-radius: 8px;
        }

        .stForm {
            background-color: #111111 !important;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🔮 Astrology Prediction System")

st.markdown("""
Enter your name, date of birth, and zodiac sign to get predictions:
- When will you get married?
- What is your personality like?
- How long might you live?
""")

# Load dataset
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("Astrology_prediction_ENG.csv")
    except:
        st.warning("Dataset not found. Please check the file name or path.")
        return pd.DataFrame()

df = load_dataset()

# Rule-based personality lookup
zodiac_to_personality = {
    "Aries": "Bold and confident",
    "Taurus": "Calm and stable",
    "Gemini": "Talkative and enthusiastic",
    "Cancer": "Sensitive and caring",
    "Leo": "Leader and proud",
    "Virgo": "Precise and organized",
    "Libra": "Balanced and artistic",
    "Scorpio": "Mysterious and intense",
    "Sagittarius": "Adventurous and wise",
    "Capricorn": "Disciplined and hardworking",
    "Aquarius": "Intellectual and independent",
    "Pisces": "Creative and empathetic"
}

if not df.empty:
    name_options = df["Name"].unique().tolist()
    zodiac_options = df["Zodiac"].unique().tolist()

    with st.form("astro_form"):
        selected_name = st.selectbox("Select Name", name_options)

        # Birthdate inputs
        birth_day = st.selectbox("Birth Day", list(range(1, 32)))
        birth_month = st.selectbox("Birth Month", [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        birth_year = st.selectbox("Birth Year", list(range(1980, datetime.now().year + 1)))

        selected_zodiac = st.selectbox("Select Zodiac", zodiac_options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Format DOB
        month_mapping = {
            "January": "01", "February": "02", "March": "03", "April": "04",
            "May": "05", "June": "06", "July": "07", "August": "08",
            "September": "09", "October": "10", "November": "11", "December": "12"
        }
        formatted_dob = f"{birth_year}-{month_mapping[birth_month]}-{str(birth_day).zfill(2)}"

        # Encode and train
        le = LabelEncoder() 
        df["Zodiac_Code"] = le.fit_transform(df["Zodiac"])
        zodiac_num = le.transform([selected_zodiac])[0]

        X = df[["Zodiac_Code"]]
        y_marriage = df["Marriage_Age"]
        y_life = df["Expected_Lifespan"]

        rf_marry = RandomForestRegressor(n_estimators=100, random_state=0)
        rf_life = RandomForestRegressor(n_estimators=100, random_state=0)
        rf_marry.fit(X, y_marriage)
        rf_life.fit(X, y_life)

        # Predict
        predicted_marriage_age = int(rf_marry.predict([[zodiac_num]])[0])
        predicted_lifespan = int(rf_life.predict([[zodiac_num]])[0])
        personality = zodiac_to_personality.get(selected_zodiac, "No information")

        # Output
        st.success(f"🔆 Prediction for {selected_name}:")
        st.write(f"💍 **Predicted Age of Marriage:** {predicted_marriage_age} years")
        st.write(f"🧠 **Personality:** {personality}")
        st.write(f"⏳ **Expected Lifespan:** {predicted_lifespan} years")
        st.write(f"📅 **Date of Birth:** {formatted_dob}")
else:
    st.error("Failed to load dataset.")
