# Diabetes Treatment Planner

An interactive web application for healthcare providers to develop personalized treatment plans for patients with diabetes or pre-diabetes. The application uses machine learning models to analyze patient data and provide comprehensive treatment recommendations.

## Features

- 🏥 Interactive patient data input
- 📊 Real-time risk assessment
- 💊 Personalized treatment recommendations
- 📈 Visual health metrics analysis
- ⚠️ Comprehensive risk factor analysis
- 📋 Detailed action items and summary

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Enter patient information in the sidebar:
   - Basic health metrics (age, BMI, blood pressure, etc.)
   - Additional risk factors
   - Click "Generate Treatment Plan" to see the analysis

## Data Input Guidelines

- **Age**: 0-120 years
- **BMI**: 10-70 kg/m²
- **Blood Pressure**: 60-250 mmHg (systolic)
- **Glucose Level**: 0-500 mg/dL
- **HDL**: 0-200 mg/dL
- **Triglycerides**: 0-1000 mg/dL

## Features Description

### 1. Health Metrics Dashboard
- Interactive gauge charts for key health indicators
- Color-coded risk levels
- Real-time updates

### 2. Risk Analysis
- Comprehensive risk factor assessment
- Complications probability analysis
- Visual risk indicators

### 3. Treatment Recommendations
- Personalized treatment effectiveness analysis
- Detailed action items
- Treatment confidence scores

## Technical Details

The application uses:
- Streamlit for the web interface
- Plotly for interactive visualizations
- Scikit-learn for machine learning models
- Pandas for data processing
- Custom ML models for risk assessment and treatment recommendations

## Data Privacy

This application is designed for local deployment only. No patient data is stored or transmitted to external servers.

## Support

For issues, questions, or contributions, please open an issue in the repository. 