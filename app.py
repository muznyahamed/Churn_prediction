from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load the model when the app starts
model_path = 'xgboost_churn_model.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file {model_path} not found!")

# Initialize OpenAI
llm = OpenAI(temperature=0.7)

# Create prompt template for analysis
analysis_prompt = PromptTemplate(
    input_variables=["customer_data", "prediction", "probability"],
    template="""
    Based on the following customer data and churn prediction, provide a detailed analysis:
    
    Customer Data:
    {customer_data}
    
    Prediction: {prediction}
    Churn Probability: {probability}
    
    Please provide:
    1. Key risk factors
    2. Recommended actions to retain the customer
    3. Overall assessment
    Keep the response concise and actionable.
    """
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Remove customerID if present
        if 'customerID' in data:
            del data['customerID']
        
        # Convert JSON to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all required features are present
        required_features = ['gender', 'SeniorCitizen', 'Partner', 
                           'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV',
                           'StreamingMovies', 'Contract', 'PaperlessBilling',
                           'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
        
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)

        # Format customer data for GPT analysis
        customer_summary = "\n".join([f"{k}: {v}" for k, v in data.items()])
        
        # Generate AI analysis
        prompt = analysis_prompt.format(
            customer_data=customer_summary,
            prediction="Will Churn" if prediction[0] == 1 else "Will Not Churn",
            probability=f"{probability[0][1]*100:.2f}%"
        )
        ai_analysis = llm(prompt)

        # Prepare response
        response = {
            'churn_prediction': 'Yes' if prediction[0] == 1 else 'No',
            'churn_probability': float(probability[0][1]),
            'no_churn_probability': float(probability[0][0]),
            'ai_analysis': ai_analysis
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 