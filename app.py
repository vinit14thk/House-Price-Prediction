import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "housing.csv"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def train_model():
    df = pd.read_csv("DATA_PATH")

    df_encoded = pd.get_dummies(
        df,
        columns=[
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
            "furnishingstatus"
        ],
        drop_first=True,
        dtype=int
    )

    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns.tolist()

model, feature_columns = train_model()

st.title("🏠 House Price Prediction")
st.write(
    "Estimate a house price using a machine learning model "
    "trained on housing data."
)

st.subheader("Enter House Details")

area = st.number_input(
    "Area (square feet)",
    min_value=100,
    max_value=20000,
    value=5000,
    step=100
)

bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
stories = st.selectbox("Stories", [1, 2, 3, 4])
parking = st.selectbox("Parking spaces", [0, 1, 2, 3])

mainroad = st.selectbox("Main road access", ["Yes", "No"])
guestroom = st.selectbox("Guest room", ["Yes", "No"])
basement = st.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.selectbox("Hot-water heating", ["Yes", "No"])
airconditioning = st.selectbox("Air conditioning", ["Yes", "No"])
prefarea = st.selectbox("Preferred area", ["Yes", "No"])

furnishingstatus = st.selectbox(
    "Furnishing status",
    ["Furnished", "Semi-furnished", "Unfurnished"]
)

if st.button("Predict House Price", type="primary"):

    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad_yes": int(mainroad == "Yes"),
        "guestroom_yes": int(guestroom == "Yes"),
        "basement_yes": int(basement == "Yes"),
        "hotwaterheating_yes": int(hotwaterheating == "Yes"),
        "airconditioning_yes": int(airconditioning == "Yes"),
        "prefarea_yes": int(prefarea == "Yes"),
        "furnishingstatus_semi-furnished": int(
            furnishingstatus == "Semi-furnished"
        ),
        "furnishingstatus_unfurnished": int(
            furnishingstatus == "Unfurnished"
        )
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_columns]

    predicted_price = model.predict(input_df)[0]

    st.success("Prediction complete!")
    st.metric(
        label="Estimated House Price",
        value=f"{predicted_price:,.0f} currency units"
    )

    st.caption(
        "This is a model-generated estimate, not a guaranteed "
        "market valuation. Predictions may be inaccurate."
    )