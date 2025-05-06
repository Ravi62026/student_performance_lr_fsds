import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://cohort:cohort@cluster0.nxmkmie.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))

db = client["student"]

collection = db["student_performance"]


from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():
    with open('student_lr_model.pkl', 'rb') as f:
        model, scalar, le = pickle.load(f)
        
    return model, scalar, le

# x = load_model()
# print(x)

def preprocessing_input_data(data, scalar, le):
    data["Extracurricular Activities"] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scalar.transform(df)
    return df_transformed

def predict_data(data):
    
    model, scalar, le = load_model()
    processed_data = preprocessing_input_data(data, scalar, le)
    prediction = model.predict(processed_data)
    return prediction

def save_to_mongodb(user_data, prediction):
    # Convert user_data values to native Python types
    processed_user_data = {}
    for key, value in user_data.items():
        if isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
            processed_user_data[key] = value.item()
        else:
            processed_user_data[key] = value
    
    # Create a document to insert
    document = {
        "user_data": processed_user_data,
        "prediction": float(prediction[0]),
        "timestamp": datetime.datetime.now()
    }
    
    # Insert the document into the collection
    result = collection.insert_one(document)
    
    return result.inserted_id

def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance")
    
    hour_studied = st.number_input("Hours studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("Previous score", min_value=40, max_value=100, value=70)
    extra = st.selectbox("Extracurricular activities", ['Yes', "No"])
    sleeping_hour = st.number_input("Sleeping hours", min_value=4, max_value=10, value=7)
    number_of_paper_solved = st.number_input("Number of question papers solved", min_value=0, max_value=10, value=5)
    
    if st.button("Predict your score"):
        user_data = {
            "Hours Studied": hour_studied,
            "Previous Scores": previous_score,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleeping_hour,
            "Sample Question Papers Practiced": number_of_paper_solved
        }
        prediction = predict_data(user_data)
        
        # Save the data to MongoDB
        inserted_id = save_to_mongodb(user_data, prediction)
        
        st.success(f"Your prediction result is {prediction[0]:.2f}")
        st.info(f"Data saved to MongoDB with ID: {inserted_id}")

if __name__ == "__main__":
    main()
    