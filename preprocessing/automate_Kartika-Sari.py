import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import time

class DataPreprocessing:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data

    def preprocess(self):
        print(f"Starting preprocessing data")
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        self.data.drop(['customer_id'],axis=1,inplace=True)
        self.data['country'] = self.data['country'].map({'France': 0, 'Spain' : 1,'Germany':2})
        self.data['gender'] = self.data['gender'].map({'Male': 0, 'Female' : 1})
        self.data['balance']=self.data['balance'].astype(int)
        self.data['estimated_salary']=self.data['estimated_salary'].astype(int)

        column = self.data.select_dtypes(include=['int64', 'float64']).columns
        column = column.drop('churn')

        scaler = StandardScaler()
        self.data[column] = scaler.fit_transform(self.data[column])

    def save(self, filename: str):
        print(f"Saving preprocessed data to {filename}")
        self.data.to_csv(filename, index=False)
        

if __name__ == "__main__":
    input = os.path.join(os.path.dirname(__file__), "../Bank Customer Churn Prediction.csv")
    output = os.path.join(os.path.dirname(__file__), "preprocessed_Bank_Customer_Churn_Prediction.csv")

    data_raw = pd.read_csv(input)

    data = DataPreprocessing(data_raw)
    data.preprocess()  
    data.save(output)
    print(f"Data saved to {output}")