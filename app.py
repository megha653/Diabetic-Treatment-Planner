import streamlit as st
import pandas as pd
import numpy as np
from ok import DiabetesTreatmentPlanner
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict
import time

# Set page config
st.set_page_config(
    page_title="Diabetes Treatment Planner",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    /* Card styling */
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #0e1117;
        margin-bottom: 20px;
    }
    
    .metric-container h3 {
        color: #0e1117;
        font-size: 1.1em;
        margin-bottom: 10px;
    }
    
    .metric-container p {
        font-size: 1.4em;
        margin: 0;
        font-weight: bold;
    }
    
    /* Risk level colors */
    .risk-high {
        color: #ff4b4b !important;
    }
    
    .risk-medium {
        color: #ffa726 !important;
    }
    
    .risk-low {
        color: #00c853 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #0e1117;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #262730;
        border-radius: 4px 4px 0 0;
        color: #ffffff;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0e1117;
        border-bottom: 2px solid #ffffff;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #262730;
        color: #ffffff !important;
        border-radius: 4px;
    }
    
    /* General text colors */
    .main-text {
        color: #ffffff;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_planner():
    """Initialize and cache the DiabetesTreatmentPlanner instance."""
    planner = DiabetesTreatmentPlanner()
    planner.preprocess_data()
    planner.train_models()
    return planner

def create_gauge_chart(value: float, title: str, min_val: float, max_val: float, 
                      thresholds: Dict[str, tuple]) -> go.Figure:
    """Create a gauge chart for health metrics."""
    colors = {
        'normal': {'hex': '#00C851', 'rgba': 'rgba(0, 200, 81, 0.2)'},
        'elevated': {'hex': '#ffbb33', 'rgba': 'rgba(255, 187, 51, 0.2)'},
        'high': {'hex': '#ff4444', 'rgba': 'rgba(255, 68, 68, 0.2)'}
    }
    
    # Determine the color based on thresholds
    color = colors['normal']['hex']
    for level, (lower, upper) in thresholds.items():
        if lower <= value <= upper:
            color = colors[level]['hex']
            break
    
    # Create steps for the gauge
    steps = []
    for level, (lower, upper) in thresholds.items():
        steps.append({
            'range': [lower, upper],
            'color': colors[level]['rgba']
        })
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        title={'text': title}
    ))
    
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def main():
    st.title("🏥 Diabetes Treatment Planner")
    st.markdown("""
    Welcome to the Diabetes Treatment Planner! This interactive tool helps healthcare providers 
    develop personalized treatment plans for patients with diabetes or pre-diabetes.
    """)

    # Initialize planner
    with st.spinner("Initializing treatment planner..."):
        planner = initialize_planner()

    # Create tabs for different categories of patient information
    input_tabs = st.tabs([
        "📊 Basic Metrics",
        "🩺 Clinical Measurements",
        "💉 Lab Results",
        "🏃‍♂️ Lifestyle Factors",
        "👨‍👩‍👦 Family History"
    ])
    
    # Dictionary to store all patient data
    patient_data = {}
    
    # Tab 1: Basic Metrics
    with input_tabs[0]:
        st.subheader("Basic Health Metrics")
        col1, col2 = st.columns(2)
        
        patient_data.update({
            'age': col1.number_input("Age", min_value=0, max_value=120, value=45),
            'bmi': col1.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1),
            'weight': col1.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1),
            'height': col1.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1),
            'waist_circumference': col2.number_input("Waist Circumference (cm)", min_value=40.0, max_value=200.0, value=80.0, step=0.1),
            'blood_pressure': col2.number_input("Blood Pressure (systolic)", min_value=60, max_value=250, value=120),
            'diastolic_bp': col2.number_input("Blood Pressure (diastolic)", min_value=40, max_value=150, value=80),
            'pulse_rate': col2.number_input("Pulse Rate (bpm)", min_value=40, max_value=200, value=75)
        })
    
    # Tab 2: Clinical Measurements
    with input_tabs[1]:
        st.subheader("Clinical Measurements")
        col1, col2 = st.columns(2)
        
        patient_data.update({
            'glucose_level': col1.number_input("Fasting Glucose Level (mg/dL)", min_value=0, max_value=500, value=100),
            'glucose_tolerance': col1.number_input("Glucose Tolerance Test (mg/dL)", min_value=0, max_value=500, value=140),
            'hba1c': col1.number_input("HbA1c (%)", min_value=3.0, max_value=20.0, value=5.7, step=0.1),
            'insulin': col2.number_input("Fasting Insulin (μU/mL)", min_value=0.0, max_value=100.0, value=10.0, step=0.1),
            'c_peptide': col2.number_input("C-Peptide (ng/mL)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        })
    
    # Tab 3: Lab Results
    with input_tabs[2]:
        st.subheader("Laboratory Results")
        col1, col2 = st.columns(2)
        
        patient_data.update({
            'hdl': col1.number_input("HDL Cholesterol (mg/dL)", min_value=0, max_value=200, value=50),
            'ldl': col1.number_input("LDL Cholesterol (mg/dL)", min_value=0, max_value=300, value=100),
            'total_cholesterol': col1.number_input("Total Cholesterol (mg/dL)", min_value=0, max_value=500, value=180),
            'triglycerides': col1.number_input("Triglycerides (mg/dL)", min_value=0, max_value=1000, value=150),
            'alt': col2.number_input("ALT (U/L)", min_value=0, max_value=500, value=30),
            'ast': col2.number_input("AST (U/L)", min_value=0, max_value=500, value=25),
            'creatinine': col2.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.0, step=0.1),
            'egfr': col2.number_input("eGFR (mL/min/1.73m²)", min_value=0, max_value=200, value=90)
        })
    
    # Tab 4: Lifestyle Factors
    with input_tabs[3]:
        st.subheader("Lifestyle and Health Behaviors")
        col1, col2 = st.columns(2)
        
        patient_data.update({
            'physical_activity': col1.slider("Physical Activity Level (hours/week)", 0.0, 40.0, 3.0, 0.5),
            'sleep_hours': col1.slider("Average Sleep (hours/day)", 0.0, 16.0, 7.0, 0.5),
            'stress_level': col1.slider("Stress Level (1-10)", 1, 10, 5),
            'smoking_status': col1.selectbox("Smoking Status", 
                ["Never", "Former", "Current - Light", "Current - Heavy"]),
            'alcohol_consumption': col2.selectbox("Alcohol Consumption",
                ["None", "Occasional", "Moderate", "Heavy"]),
            'diet_adherence': col2.slider("Diet Adherence (1-10)", 1, 10, 7),
            'medication_adherence': col2.slider("Medication Adherence (1-10)", 1, 10, 8)
        })
        
        st.subheader("Current Medications")
        med_col1, med_col2, med_col3 = st.columns(3)
        
        patient_data.update({
            'on_insulin': med_col1.checkbox("Insulin"),
            'on_metformin': med_col1.checkbox("Metformin"),
            'on_sulfonylureas': med_col2.checkbox("Sulfonylureas"),
            'on_dpp4': med_col2.checkbox("DPP-4 Inhibitors"),
            'on_glp1': med_col3.checkbox("GLP-1 Receptor Agonists"),
            'on_sglt2': med_col3.checkbox("SGLT2 Inhibitors")
        })
    
    # Tab 5: Family History and Risk Factors
    with input_tabs[4]:
        st.subheader("Family History")
        col1, col2 = st.columns(2)
        
        patient_data.update({
            'family_diabetes': col1.selectbox("Family History of Diabetes",
                ["None", "One Parent", "Both Parents", "Sibling", "Multiple Family Members"]),
            'family_heart_disease': col1.checkbox("Family History of Heart Disease"),
            'family_hypertension': col1.checkbox("Family History of Hypertension"),
            'ethnicity': col2.selectbox("Ethnicity", [
                "Caucasian", "African American", "Hispanic", "Asian", "Native American", "Other"
            ])
        })
        
        st.subheader("Medical History and Risk Factors")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        patient_data.update({
            'autoimmune_conditions': risk_col1.checkbox("Autoimmune Conditions"),
            'cardiovascular_disease': risk_col1.checkbox("Cardiovascular Disease"),
            'hypertension': risk_col1.checkbox("Hypertension"),
            'gestational_diabetes': risk_col2.checkbox("History of Gestational Diabetes"),
            'pcos': risk_col2.checkbox("PCOS"),
            'thyroid_disorder': risk_col2.checkbox("Thyroid Disorder"),
            'kidney_disease': risk_col3.checkbox("Kidney Disease"),
            'liver_disease': risk_col3.checkbox("Liver Disease")
        })

    # Generate Treatment Plan button
    if st.button("🔄 Generate Treatment Plan", type="primary", use_container_width=True):
        with st.spinner("Analyzing patient data and generating treatment plan..."):
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Generate treatment plan
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            treatment_plan = planner.generate_treatment_plan(patient_data)
            
            # Remove progress bar
            progress_bar.empty()

            # Display results in main area
            st.header("📊 Analysis Results")
            
            # Create three columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            # Risk Level with colored text
            risk_prob = treatment_plan['risk_probability']
            risk_class = 'risk-high' if risk_prob > 0.7 else 'risk-medium' if risk_prob > 0.3 else 'risk-low'
            col1.markdown(f"""
                <div class="metric-container">
                    <h3>Risk Level</h3>
                    <p class="{risk_class}">{risk_prob:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Diabetes Type
            col2.markdown(f"""
                <div class="metric-container">
                    <h3>Diabetes Type</h3>
                    <p><strong>{treatment_plan['diabetes_type']}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Treatment Confidence
            # confidence = max(treatment_plan['recommended_treatments'].values())
            # col3.markdown(f"""
            #     <div class="metric-container">
            #         <h3>Treatment Confidence</h3>
            #         <p><strong>{confidence:.1%}</strong></p>
            #     </div>
            #     """, unsafe_allow_html=True)

            # Create tabs for different sections
            result_tabs = st.tabs([
                " Health Metrics",
                " Risk Analysis",
                " Treatment Plan",
                " Detailed Report"
            ])
            
            with result_tabs[0]:
                st.subheader("Key Health Metrics")
                
                # Create gauge charts for key metrics
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    # Glucose Level gauge
                    glucose_thresholds = {
                        'normal': (70, 99),
                        'elevated': (100, 125),
                        'high': (126, 200)
                    }
                    st.plotly_chart(create_gauge_chart(
                        patient_data['glucose_level'],
                        "Glucose Level (mg/dL)",
                        0, 200,
                        glucose_thresholds
                    ), use_container_width=True)
                    
                    # BMI gauge
                    bmi_thresholds = {
                        'normal': (18.5, 24.9),
                        'elevated': (25, 29.9),
                        'high': (30, 40)
                    }
                    st.plotly_chart(create_gauge_chart(
                        patient_data['bmi'],
                        "BMI",
                        15, 40,
                        bmi_thresholds
                    ), use_container_width=True)
                
                with metrics_col2:
                    # Blood Pressure gauge
                    bp_thresholds = {
                        'normal': (90, 120),
                        'elevated': (121, 130),
                        'high': (131, 180)
                    }
                    st.plotly_chart(create_gauge_chart(
                        patient_data['blood_pressure'],
                        "Blood Pressure (systolic)",
                        80, 200,
                        bp_thresholds
                    ), use_container_width=True)
                    
                    # HbA1c gauge
                    hba1c_thresholds = {
                        'normal': (4.0, 5.6),
                        'elevated': (5.7, 6.4),
                        'high': (6.5, 10.0)
                    }
                    st.plotly_chart(create_gauge_chart(
                        patient_data['hba1c'],
                        "HbA1c (%)",
                        3, 12,
                        hba1c_thresholds
                    ), use_container_width=True)
            
            with result_tabs[1]:
                st.subheader("Risk Factor Analysis")
                
                # Create two columns for risk factors and complications
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    st.markdown("### Key Risk Factors")
                    for factor, data in treatment_plan['risk_factors'].items():
                        status_color = (
                            "🔴" if data['status'] in ['High', 'Obese'] else 
                            "🟡" if data['status'] in ['Elevated', 'Overweight'] else 
                            "🟢"
                        )
                        st.markdown(f"**{factor.title()}**: {status_color} {data['status']}")
                
                # with risk_col2:
                #     st.markdown("### Treatment Confidence")
                #     for treatment, confidence in treatment_plan['recommended_treatments'].items():
                #         conf_color = "🔴" if confidence < 0.3 else "🟡" if confidence < 0.7 else "🟢"
                #         st.markdown(f"**{treatment.replace('_', ' ').title()}**: {conf_color} {confidence:.1%}")
            
            with result_tabs[2]:
                st.subheader("Treatment Recommendations")
                
                # Display treatment effectiveness
                st.markdown("### 💊 Treatment Effectiveness")
                effectiveness_data = pd.DataFrame({
                    'Treatment': [t.replace('_', ' ').title() for t in treatment_plan['recommended_treatments'].keys()],
                    'Effectiveness': list(treatment_plan['recommended_treatments'].values())
                })
                
                fig = px.bar(effectiveness_data,
                           x='Treatment',
                           y='Effectiveness',
                           color='Effectiveness',
                           color_continuous_scale=['red', 'yellow', 'green'],
                           labels={'Effectiveness': 'Expected Effectiveness'})
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display recommendations
                st.markdown("### 📋 Action Items")
                for i, recommendation in enumerate(treatment_plan['recommendations'], 1):
                    st.markdown(f"{i}. {recommendation}")
            
            with result_tabs[3]:
                st.subheader("Detailed Patient Report")
                
                # Create expandable sections for different aspects of the report
                with st.expander("🔬 Laboratory Analysis", expanded=True):
                    lab_col1, lab_col2 = st.columns(2)
                    with lab_col1:
                        st.markdown("#### Glucose Metabolism")
                        st.markdown(f"- Fasting Glucose: **{patient_data['glucose_level']} mg/dL**")
                        st.markdown(f"- HbA1c: **{patient_data['hba1c']}%**")
                        st.markdown(f"- Glucose Tolerance: **{patient_data['glucose_tolerance']} mg/dL**")
                        st.markdown(f"- Insulin: **{patient_data['insulin']} μU/mL**")
                    
                    with lab_col2:
                        st.markdown("#### Lipid Profile")
                        st.markdown(f"- Total Cholesterol: **{patient_data['total_cholesterol']} mg/dL**")
                        st.markdown(f"- HDL: **{patient_data['hdl']} mg/dL**")
                        st.markdown(f"- LDL: **{patient_data['ldl']} mg/dL**")
                        st.markdown(f"- Triglycerides: **{patient_data['triglycerides']} mg/dL**")
                
                with st.expander("❤️ Cardiovascular Status"):
                    cv_col1, cv_col2 = st.columns(2)
                    with cv_col1:
                        st.markdown(f"- Systolic BP: **{patient_data['blood_pressure']} mmHg**")
                        st.markdown(f"- Diastolic BP: **{patient_data['diastolic_bp']} mmHg**")
                        st.markdown(f"- Pulse Rate: **{patient_data['pulse_rate']} bpm**")
                    with cv_col2:
                        st.markdown("#### Risk Factors")
                        st.markdown("- Hypertension: " + ("Yes ⚠️" if patient_data['hypertension'] else "No ✓"))
                        st.markdown("- Family History: " + ("Yes ⚠️" if patient_data['family_heart_disease'] else "No ✓"))
                
                with st.expander("🏃‍♂️ Lifestyle Assessment"):
                    lifestyle_col1, lifestyle_col2 = st.columns(2)
                    with lifestyle_col1:
                        st.markdown(f"- Physical Activity: **{patient_data['physical_activity']} hours/week**")
                        st.markdown(f"- Sleep: **{patient_data['sleep_hours']} hours/day**")
                        st.markdown(f"- Stress Level: **{patient_data['stress_level']}/10**")
                    with lifestyle_col2:
                        st.markdown(f"- Smoking: **{patient_data['smoking_status']}**")
                        st.markdown(f"- Alcohol: **{patient_data['alcohol_consumption']}**")
                        st.markdown(f"- Diet Adherence: **{patient_data['diet_adherence']}/10**")
                
                # Summary statistics
                st.markdown("### 📝 Summary")
                summary = treatment_plan['summary']
                st.markdown(f"- **Total Recommendations**: {summary['total_recommendations']}")
                if summary['key_areas_of_concern']:
                    st.markdown(f"- **Key Areas of Concern**: {', '.join(summary['key_areas_of_concern'])}")
                if summary['primary_treatments']:
                    st.markdown(f"- **Primary Treatments**: {', '.join(summary['primary_treatments'])}")

if __name__ == "__main__":
    main()
