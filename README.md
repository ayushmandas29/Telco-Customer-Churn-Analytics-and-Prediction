![GitHub repo license](https://img.shields.io/badge/license-MIT-green?style=flat-square&logo=github)
![Python version](https://img.shields.io/badge/python-3.10-blue?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-yes-brightgreen?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)

# 📊 Telco Customer Churn Analytics & Prediction | Streamlit Dashboard

Live deployed ML dashboard for analyzing telecom customer churn and predicting whether a customer is likely to leave the service.

🔗 **Live App:**  
https://telco-customer-churn-analytics-and-prediction-no2yxxrc4mnzgyun.streamlit.app/


## 🚀 What this project does
- Visualizes churn patterns and key business KPIs
- Shows customer segments most likely to churn
- Predicts churn likelihood using a trained ML model
- Allows users to simulate “what-if” churn scenarios



## 🏗 Tech Stack
| Category | Tools |
|---------|-------|
| Language | Python |
| ML | Scikit-Learn |
| Data | Pandas, NumPy |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Version Control | Git + GitHub |
| Deployment | Streamlit Cloud |



## 📂 Project Structure
Telco-Customer-Churn-Analytics-and-Prediction/
│
├── dashboard/
│ └── app.py # Streamlit dashboard
├── data/
│ └── processed/
│ └── telco_churn_clean.csv
├── requirements.txt
└── README.md


## 📌 Key Features
✔ KPI cards: churn rate, customer count, monthly charges  
✔ Interactive filters (gender, contract type, tenure, etc.)  
✔ Visual insights using Plotly charts  
✔ Churn prediction form powered by trained ML model  
✔ Fully deployed and accessible via browser — no installation required

---

## 🔄 Machine Learning (high-level overview)
- Encoded categorical variables and cleaned telecom dataset
- Train/test split performed
- Random Forest model trained for churn classification
- Evaluated using accuracy, F1-score and confusion matrix

---

## 💻 Run Locally (for developers)
```bash
git clone https://github.com/ayushmandas29/Telco-Customer-Churn-Analytics-and-Prediction.git
cd Telco-Customer-Churn-Analytics-and-Prediction
pip install -r requirements.txt
streamlit run dashboard/app.py
🔧 Future Enhancements (roadmap)
Add PostgreSQL database integration for storing customer records

Improve churn prediction with multiple ML models (XGBoost, Logistic Regression)

Add SHAP explainability for feature impact analysis

Add authentication / user login

Power BI integration for BI reporting

👤 Author
Ayushman Das
GitHub: https://github.com/ayushmandas29

⭐ If this project helped you, consider starring the repo!
