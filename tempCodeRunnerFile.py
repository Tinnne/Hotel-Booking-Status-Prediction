import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

class LoanPredictionInference:
    def __init__(self, model_path, input_data_path):
        self.model_path = model_path
        self.input_data_path = input_data_path
        self.scaler = RobustScaler()
        self.model = None
        self.data = None

    def load_all(self):
        self.model = joblib.load(self.model_path)
        self.data = pd.read_csv(self.input_data_path)
        print("Model and Data loaded successfully.")

    def preprocess(self):
        self.data = self.data.drop(columns=['Booking_ID'])

        self.data['arrival_year'] = self.data['arrival_year'].replace({2017: 0, 2018: 1})   
        self.data['booking_status'] = self.data['booking_status'].replace({'Canceled': 1, 'Not_Canceled': 0})
        
        self.data = pd.get_dummies(self.data, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'])

    def predict(self):
        predictions = self.model.predict(self.data)
        print("Predictions complete.")
        return predictions

if __name__ == "__main__":
    model_path = "xgb_model.pkl"
    input_data_path = "data/Dataset_b_hotel.csv"

    inference = LoanPredictionInference(model_path, input_data_path)
    inference.load_all()
    inference.preprocess()
    preds = inference.predict()

    print("Predicted Booking Status (0 = Not Canceled, 1 = Canceled):")
    print(preds)
