# 📊 Telco Customer Churn Analytics & Prediction (Streamlit Dashboard)

An end-to-end **customer churn analytics and prediction** project built using  
**Python, Machine Learning and an interactive Streamlit dashboard**, deployed on the cloud.

---

## 🚀 Live Demo

🔗 **Streamlit App:**  
https://telco-customer-churn-analytics-and-prediction-no2yxxrc4mnzgyun.streamlit.app/

You can open this link in any browser and use the dashboard without installing anything.

---

## 🧠 Project Overview

Telecom companies lose money when customers **churn** (leave the service).  
This project simulates a real telecom scenario:

- Cleans and processes telco customer data  
- Analyzes patterns related to churn (tenure, contract, charges, etc.)  
- Trains a Machine Learning model to predict if a customer will churn  
- Exposes insights and predictions through a **web dashboard** built in Streamlit  

This is designed as a **portfolio / industry-style project**, not just a basic assignment.

---

## 🏗 Tech Stack

- **Language:** Python  
- **Data / ML:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Plotly, Matplotlib / Seaborn  
- **App / UI:** Streamlit  
- **Version Control:** Git & GitHub  
- **Deployment:** Streamlit Community Cloud  

---

## 📁 Project Structure

```bash
Telco-Customer-Churn-Analytics-and-Prediction/
├── dashboard/
│   └── app.py                     # Main Streamlit dashboard
├── data/
│   └── processed/
│       └── telco_churn_clean.csv  # Cleaned dataset used by the app
├── reports/
│   └── figures/                   # Plots / figures (optional)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation

✨ Key Features

📈 KPIs & Summary

Total customers

Overall churn rate

Average monthly charges

📊 Interactive Analytics

Churn by contract type

Churn vs tenure

Churn vs monthly charges

Filterable customer records

🤖 Churn Prediction (What-if)

Simple form where you enter customer details (tenure, internet, contract, charges, etc.)

ML model predicts whether the customer is likely to churn or not

🌐 Deployed Dashboard

Hosted on Streamlit Cloud

Can be accessed from any device with a browser

🔄 Machine Learning (High Level)

Data Cleaning & Preprocessing

Handle missing values

Convert categorical variables to numeric encodings

Ensure correct data types for numeric columns

Train / Test Split

Split data into training and test sets

Model

Train a classification model (e.g. Random Forest)

Evaluate using accuracy, F1-score, classification report, confusion matrix

Integration with Dashboard

The trained model is integrated into the Streamlit app

User input from the UI is passed to the model for prediction

💻 How to Run Locally
1️⃣ Clone this repo
git clone https://github.com/ayushmandas29/Telco-Customer-Churn-Analytics-and-Prediction.git
cd Telco-Customer-Churn-Analytics-and-Prediction

2️⃣ (Optional) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On macOS/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run dashboard/app.py


Open the URL shown in the terminal (usually http://localhost:8501).

🚀 Deployment

The app is deployed using Streamlit Community Cloud:

Code is hosted on GitHub

Streamlit pulls this repo and runs dashboard/app.py

requirements.txt is used to install dependencies on the server

Any future git push to main can trigger a new deployment.

📌 Future Improvements

Compare multiple ML models (Logistic Regression, XGBoost, etc.)

Add model explainability (feature importance / SHAP)

Store customer data and predictions in a real database (PostgreSQL)

Add authentication / login for internal dashboards

Build a Power BI version of the same churn analysis

👤 Author

Ayushman Das
GitHub: @ayushmandas29

If you find this useful, feel free to ⭐ star the repo!
