import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

class LoanPredictionModel:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = RobustScaler()
        self.data = pd.read_csv(self.filepath)
        self.train_data = None
        self.test_data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.num_cols = ["lead_time", "avg_price_per_room"]

    def clean_data(self):
        self.data.drop_duplicates(inplace=True)
        self.data.dropna(inplace=True)
        print("Data cleaning complete")

    def encode_columns(self):
        self.data = pd.get_dummies(self.data, columns=['type_of_meal_plan', 'room_type_reserved', "market_segment_type"])

        self.data['arrival_year'] = self.data['arrival_year'].replace({2017: 0, 2018: 1})
        self.data['booking_status'] = self.data['booking_status'].replace({'Canceled': 1, 'Not_Canceled': 0})

    def split_data(self, test_size=0.2, random_state=42):
        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)

    def scale_numeric_columns(self):
        self.train_data[self.num_cols] = self.scaler.fit_transform(self.train_data[self.num_cols])
        self.test_data[self.num_cols] = self.scaler.transform(self.test_data[self.num_cols])

    def define_features_labels(self):
        self.x_train = self.train_data.drop(columns=["Booking_ID", "booking_status"])
        self.y_train = self.train_data["booking_status"]

        self.x_test = self.test_data.drop(columns=["Booking_ID", "booking_status"])
        self.y_test = self.test_data["booking_status"]

    def preprocess_data(self):
        self.encode_columns()
        self.split_data()
        self.scale_numeric_columns()
        self.define_features_labels()
        print("Preprocessing complete")

    def train_model(self):
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.x_train, self.y_train)
        print("Model training complete")

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test)

        print("Model Evaluation:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(self.y_test, y_pred, average='weighted'):.4f}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred, average='weighted'):.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

    def run(self):
        self.clean_data()
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()


if __name__ == "__main__":
    model = LoanPredictionModel("Data/Dataset_b_hotel.csv")
    model.run()