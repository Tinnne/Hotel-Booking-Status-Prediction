import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

class LoanPredictionInference:
    def __init__(self, model_path, input_data_path, features_path, scaler_path):
        self.model_path = model_path
        self.input_data_path = input_data_path
        self.features_path = features_path
        self.scaler_path = scaler_path

        self.model = None
        self.data = None
        self.feature_columns = None
        self.scaler = None

    def load_all(self):
        self.model = joblib.load(self.model_path)
        self.data = pd.read_csv(self.input_data_path)
        self.feature_columns = joblib.load(self.features_path)
        self.scaler = joblib.load(self.scaler_path)
        print("Model and Data loaded successfully.")

    def preprocess(self):
        if 'Booking_ID' in self.data.columns:
            self.data = self.data.drop(columns=['Booking_ID'])
        if 'booking_status' in self.data.columns:
            self.data = self.data.drop(columns=['booking_status'])

        self.data['arrival_year'] = self.data['arrival_year'].replace({2017: 0, 2018: 1})   
        
        self.data = pd.get_dummies(self.data, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'])

        if self.feature_columns:
            self.data = self.data.reindex(columns=self.feature_columns, fill_value=0)

        num_cols = ["lead_time", "avg_price_per_room"]
        self.data[num_cols] = self.scaler.transform(self.data[num_cols])

    def predict(self):
        predictions = self.model.predict(self.data)
        print("Predictions complete.")
        return predictions

if __name__ == "__main__":
    model_path = "Model/xgb_model.pkl"
    input_data_path = "Data/test.csv"
    features_path = "Model/model_features.pkl"
    scaler_path = "Model/scaler.pkl"

    inference = LoanPredictionInference(model_path, input_data_path, features_path, scaler_path)
    inference.load_all()
    inference.preprocess()
    preds = inference.predict()

    print("Predicted Booking Status (0 = Not Canceled, 1 = Canceled):")
    print(preds)
