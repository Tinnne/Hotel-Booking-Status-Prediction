import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("Model/xgb_model.pkl")
    features = joblib.load("Model/model_features.pkl")
    scaler = joblib.load("Model/scaler.pkl")
    return model, features, scaler

model, features, scaler = load_artifacts()

st.title("🏨 Hotel Booking Status Predictor")

st.sidebar.title("Choose Input Method")
input_method = st.sidebar.radio("Input Type", ["Manual Entry", "Upload CSV"])

# --- Manual Entry Mode ---
if input_method == "Manual Entry":
    example_cases = {
    "Example 1: Likely Not Canceled": {
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
        "required_car_parking_space": 1,
        "lead_time": 10,
        "arrival_year": 2018,
        "arrival_month": 5,
        "arrival_date": 12,
        "repeated_guest": 1,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 2,
        "avg_price_per_room": 80.0,
        "no_of_special_requests": 1,
        "type_of_meal_plan": 'Meal Plan 1',
        "room_type_reserved": 'Room_Type 1',
        "market_segment_type": 'Online',
    },
    "Example 2: Likely Canceled": {
        "no_of_adults": 1,
        "no_of_children": 2,
        "no_of_weekend_nights": 2,
        "no_of_week_nights": 3,
        "required_car_parking_space": 0,
        "lead_time": 300,
        "arrival_year": 2017,
        "arrival_month": 11,
        "arrival_date": 24,
        "repeated_guest": 0,
        "no_of_previous_cancellations": 2,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 200.0,
        "no_of_special_requests": 0,
        "type_of_meal_plan": 'Not Selected',
        "room_type_reserved": 'Room_Type 4',
        "market_segment_type": 'Offline',
    }
}

    selected_example = st.sidebar.selectbox("Or select an example case", ["None"] + list(example_cases.keys()))

    if selected_example != "None":
        example = example_cases[selected_example]
        no_of_adults = example["no_of_adults"]
        no_of_children = example["no_of_children"]
        no_of_weekend_nights = example["no_of_weekend_nights"]
        no_of_week_nights = example["no_of_week_nights"]
        required_car_parking_space = example["required_car_parking_space"]
        lead_time = example["lead_time"]
        arrival_year = example["arrival_year"]
        arrival_month = example["arrival_month"]
        arrival_date = example["arrival_date"]
        repeated_guest = example["repeated_guest"]
        no_of_previous_cancellations = example["no_of_previous_cancellations"]
        no_of_previous_bookings_not_canceled = example["no_of_previous_bookings_not_canceled"]
        avg_price_per_room = example["avg_price_per_room"]
        no_of_special_requests = example["no_of_special_requests"]
        type_of_meal_plan = example["type_of_meal_plan"]
        room_type_reserved = example["room_type_reserved"]
        market_segment_type = example["market_segment_type"]


    st.subheader("📝 Enter Booking Info")

    # The inputs
        # Use example values if selected, otherwise use defaults
    example = example_cases[selected_example] if selected_example != "None" else {}

    no_of_adults = st.number_input("No. of Adults", 1, 10, example.get("no_of_adults", 2))
    no_of_children = st.number_input("No. of Children", 0, 10, example.get("no_of_children", 0))
    no_of_weekend_nights = st.number_input("Weekend Nights", 0, 10, example.get("no_of_weekend_nights", 1))
    no_of_week_nights = st.number_input("Week Nights", 0, 20, example.get("no_of_week_nights", 2))
    required_car_parking_space = st.selectbox("Parking Space Required", [0, 1], index=[0, 1].index(example.get("required_car_parking_space", 0)))
    lead_time = st.slider("Lead Time (days)", 0, 500, example.get("lead_time", 100))
    arrival_year = st.selectbox("Arrival Year", [2017, 2018], index=[2017, 2018].index(example.get("arrival_year", 2017)))
    arrival_month = st.slider("Arrival Month", 1, 12, example.get("arrival_month", 6))
    arrival_date = st.slider("Arrival Date", 1, 31, example.get("arrival_date", 15))
    repeated_guest = st.selectbox("Repeated Guest", [0, 1], index=[0, 1].index(example.get("repeated_guest", 0)))
    no_of_previous_cancellations = st.number_input("Previous Cancellations", 0, 10, example.get("no_of_previous_cancellations", 0))
    no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", 0, 10, example.get("no_of_previous_bookings_not_canceled", 0))
    avg_price_per_room = st.slider("Avg Price per Room", 0.0, 500.0, example.get("avg_price_per_room", 100.0))
    no_of_special_requests = st.number_input("Special Requests", 0, 5, example.get("no_of_special_requests", 0))

    type_of_meal_plan = st.selectbox("Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'],
                                     index=['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'].index(example.get("type_of_meal_plan", 'Meal Plan 1')))
    room_type_reserved = st.selectbox("Room Type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'],
                                      index=['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'].index(example.get("room_type_reserved", 'Room_Type 1')))
    market_segment_type = st.selectbox("Market Segment", ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary'],
                                       index=['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary'].index(example.get("market_segment_type", 'Online')))


    if st.button("Predict"):
        input_dict = {
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "required_car_parking_space": required_car_parking_space,
            "lead_time": lead_time,
            "arrival_year": 0 if arrival_year == 2017 else 1,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests,
        }

        df_input = pd.DataFrame([input_dict])
        df_input = pd.get_dummies(df_input, columns=[])
        df_input[f'type_of_meal_plan_{type_of_meal_plan}'] = 1
        df_input[f'room_type_reserved_{room_type_reserved}'] = 1
        df_input[f'market_segment_type_{market_segment_type}'] = 1

        df_input = df_input.reindex(columns=features, fill_value=0)
        df_input[["lead_time", "avg_price_per_room"]] = scaler.transform(df_input[["lead_time", "avg_price_per_room"]])

        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        st.success(f"Prediction: {'Canceled' if prediction == 1 else 'Not Canceled'} (Canceled Probability: {probability:.2f})")

# --- CSV Upload Mode ---
elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Uploaded Data", df.head())

        # Preserve Booking_ID before modifying df
        booking_ids = df['Booking_ID'] if 'Booking_ID' in df.columns else pd.Series(range(len(df)))

        # Drop unneeded columns
        if 'Booking_ID' in df.columns:
            df.drop(columns=['Booking_ID'], inplace=True)
        if 'booking_status' in df.columns:
            df.drop(columns=['booking_status'], inplace=True)

        # Preprocessing
        df['arrival_year'] = df['arrival_year'].replace({2017: 0, 2018: 1})
        df = pd.get_dummies(df, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'])
        df = df.reindex(columns=features, fill_value=0)

        num_cols = ["lead_time", "avg_price_per_room"]
        df[num_cols] = scaler.transform(df[num_cols])

        # Predict
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        # Display results in a table
        results_df = pd.DataFrame({
            'Booking_ID': booking_ids,
            'Probability (Canceled)': probs,
            'Prediction': preds
        })
        results_df['Prediction Label'] = results_df['Prediction'].map({0: "Not Canceled", 1: "Canceled"})

        st.write("### Predictions with Probabilities")
        st.dataframe(results_df)

        # Visualization
        canceled = np.sum(preds)
        not_canceled = len(preds) - canceled
        fig, ax = plt.subplots()
        ax.pie([not_canceled, canceled], labels=["Not Canceled", "Canceled"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

# use:
# streamlit run Number4.py
# to run the file locally
