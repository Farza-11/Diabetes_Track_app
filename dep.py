import streamlit as st
import pandas as pd
import joblib

# Loading the saved model
model = joblib.load('RandomForest_Model.pkl')  # Make sure the model file exists in the correct location

def predict_diabetes(data):
    prediction = model.predict(data)
    return prediction

def main():
    
    st.markdown(
        """
       <style> 
             .title {{
             color: #0000FF
             text-align: center;
             font-size: 26px;
             padding-top: 30px;
             margin-bottom: 50px;
             }}
       </style>
""",
unsafe_allow_html=True
)

st.title("Diabetes Prediction App " )  # Moved this title to the top
st.markdown(
    """
    <style>
    .subtitle {
        text-align: center;
        font-size: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title('Values to predict Diabetes: ')
st.markdown("Enter the values to predict the Diabetes:")
age = st.number_input("Age", min_value=0, max_value=130, step=1, value=30)
hypertension = st.radio("Hypertension", options=["Yes", "No"])
heart_disease = st.radio("Heart Disease", options=["Yes", "No"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, value=25.0)
HbAlc_level = st.number_input("HBalc_Level", min_value=0.0, max_value=30.0, step=0.1, value=5.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=10.0, max_value=300.0, step=1.0, value=50.0)

# Adjusted column names and data type conversion in the input data DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'hypertension': [1 if hypertension == 'Yes' else 0],  # Convert 'Yes' to 1 and 'No' to 0
    'heart_disease': [1 if heart_disease == 'Yes' else 0],  # Convert 'Yes' to 1 and 'No' to 0
    'bmi': [bmi],
    'HbA1c_level': [HbAlc_level],
    'blood_glucose_level': [blood_glucose_level],
})

# Predict button
if st.button("Predict", key="predict_button"):
    result = predict_diabetes(input_data)
    if result[0] == 0:
        st.markdown(
            """
        <div style="background-color:#f9f9f9;padding:0px;border-radius:20px;">
                    <h3 style="color:#f63366;text-align:center;">Results:</h3>
                    <p style="text-align:center;font-size:24px;">You are healthy.</p>
                </div>
                """,
            unsafe_allow_html=True
        )
    elif result[0] == 1:
        st.markdown(
            """
                <div style="background-color:#f9f9f9;padding:0px;border-radius:20px;">
                    <h3 style="color:#f63366;text-align:center;">Results:</h3>
                    <p style="text-align:center;font-size:24px;">You are diabetic.</p>
                </div>
                """,
            unsafe_allow_html=True
        )


if __name__ == '__main__':
    main()
    st.write("-by Farza Haider")
