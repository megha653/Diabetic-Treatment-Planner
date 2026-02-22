import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DiabetesTreatmentPlanner:
    def __init__(self, data_path: str = 'nhanes/'):
        """Initialize the diabetes treatment planner with NHANES dataset."""
        self.data_path = data_path
        self.complications = ['retinopathy', 'nephropathy', 'neuropathy', 'cardiovascular']
        
        # Function to safely read CSV with different encodings
        def safe_read_csv(file_path):
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, nrows=5)
                    # If successful, read the full file
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
                except FileNotFoundError:
                    print(f"Warning: File {file_path} not found.")
                    return pd.DataFrame()
            print(f"Warning: Could not read {file_path} with any of the attempted encodings.")
            return pd.DataFrame()

        # Read all data files
        print("Loading datasets...")
        self.demographic_data = safe_read_csv(f'{data_path}demographic.csv')
        self.examination_data = safe_read_csv(f'{data_path}examination.csv')
        self.labs_data = safe_read_csv(f'{data_path}labs.csv')
        self.medications_data = safe_read_csv(f'{data_path}medications.csv')
        self.diet_data = safe_read_csv(f'{data_path}diet.csv')
        self.questionnaire_data = safe_read_csv(f'{data_path}questionnaire.csv')

        # Print available columns for debugging
        print("\nAvailable columns in datasets:")
        for name, df in [
            ('Demographics', self.demographic_data),
            ('Examination', self.examination_data),
            ('Labs', self.labs_data),
            ('Medications', self.medications_data),
            ('Diet', self.diet_data),
            ('Questionnaire', self.questionnaire_data)
        ]:
            if not df.empty:
                print(f"\n{name} columns:", df.columns.tolist())

    def print_section_header(self, title):
        """Print a formatted section header."""
        print(f"\n{'='*80}")
        print(f"{title.center(80)}")
        print(f"{'='*80}\n")

    def preprocess_data(self):
        """Preprocess and merge relevant data for analysis."""
        self.print_section_header("Data Preprocessing")
        
        if not self.demographic_data.empty:
            self.merged_data = self.demographic_data
        else:
            print("❌ Error: No demographic data available.")
            return

        # Function to safely merge dataframes
        def safe_merge(left_df, right_df, common_cols=None):
            if right_df.empty:
                return left_df
            
            if common_cols is None:
                common_cols = ['SEQN']
            
            if not all(col in right_df.columns for col in common_cols):
                print(f"⚠️  Warning: Missing merge columns in right dataframe")
                return left_df
            
            print(f"✓ Merging datasets using: {', '.join(common_cols)}")
            return left_df.merge(right_df, on=common_cols, how='left')

        # Merge datasets
        print("\n📊 Merging Datasets:")
        self.merged_data = safe_merge(self.demographic_data, self.examination_data)
        self.merged_data = safe_merge(self.merged_data, self.labs_data)
        self.merged_data = safe_merge(self.merged_data, self.medications_data)

        # Create features
        self.print_section_header("Feature Engineering")
        
        # Map NHANES columns to our features
        nhanes_mapping = {
            'glucose_level': {
                'primary': 'LBXGLU',  # Fasting Glucose
                'secondary': ['LBXSGL', 'LBXGLT'],  # Other glucose measurements
                'default_value': 100
            },
            'blood_pressure': {
                'primary': 'BPXSY1',  # Systolic Blood Pressure
                'secondary': ['BPXSY2', 'BPXSY3', 'BPXSY4'],
                'default_value': 120
            },
            'bmi': {
                'primary': 'BMXBMI',  # Body Mass Index
                'secondary': [],
                'default_value': 25
            },
            'age': {
                'primary': 'RIDAGEYR',  # Age in years
                'secondary': [],
                'default_value': 45
            },
            'hdl': {
                'primary': 'LBDHDD',  # Direct HDL-Cholesterol
                'secondary': [],
                'default_value': 50
            },
            'triglycerides': {
                'primary': 'LBXTR',  # Triglycerides
                'secondary': ['LBXSTR'],
                'default_value': 150
            }
        }

        # Function to get value from multiple possible columns
        def get_value_from_columns(df, columns):
            for col in columns:
                if col in df.columns and not df[col].isna().all():
                    print(f"Using column {col}")
                    return df[col]
            return None

        # Create features using the mapping
        print("🔍 Processing Features:")
        for feature, mapping in nhanes_mapping.items():
            value = get_value_from_columns(self.merged_data, [mapping['primary']])
            
            if value is None:
                value = get_value_from_columns(self.merged_data, mapping['secondary'])
            
            if value is None:
                print(f"  ⚠️  Creating synthetic data for: {feature}")
                value = pd.Series(np.random.normal(
                    mapping['default_value'], 
                    mapping['default_value'] * 0.2, 
                    len(self.merged_data)
                ))
            else:
                print(f"  ✓ Successfully processed: {feature}")
            
            self.merged_data[feature] = value

        # Add diabetes-related information
        print("\n🏥 Processing Clinical Information:")
        if 'DIQ010' in self.questionnaire_data.columns:
            self.merged_data['has_diabetes'] = self.questionnaire_data['DIQ010'].map({
                1: 1,  # Yes
                2: 0,  # No
                3: 0,  # Borderline
                7: np.nan,  # Refused
                9: np.nan   # Don't know
            })
            print("  ✓ Added diabetes status from questionnaire")
        else:
            print("  ⚠️  Using glucose levels to determine diabetes status")
            self.merged_data['has_diabetes'] = (self.merged_data['glucose_level'] >= 126).astype(int)

        if 'RXDUSE' in self.medications_data.columns:
            self.merged_data['on_medication'] = self.medications_data['RXDUSE'].map({
                1: 1,  # Yes
                2: 0   # No
            })
            print("  ✓ Added medication information")
        else:
            self.merged_data['on_medication'] = 0
            print("  ⚠️  No medication information available")

        # Handle missing values
        print("\n🔧 Handling Missing Values...")
        numeric_columns = self.merged_data.select_dtypes(include=[np.number]).columns
        self.merged_data[numeric_columns] = self.merged_data[numeric_columns].fillna(
            self.merged_data[numeric_columns].mean()
        )
        print("  ✓ Filled missing values with mean values")

        # Create risk groups
        self.merged_data['risk_group'] = pd.cut(
            self.merged_data['glucose_level'],
            bins=[0, 99, 125, float('inf')],
            labels=['normal', 'pre_diabetic', 'diabetic']
        )
        print("  ✓ Created risk groups based on glucose levels")

        # Print summary statistics
        self.print_section_header("Data Summary")
        print(f"📊 Total Records: {len(self.merged_data):,}")
        print("\n📈 Feature Statistics:")
        stats = self.merged_data[list(nhanes_mapping.keys())].describe()
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(stats)

    def train_models(self):
        """Train all necessary models for comprehensive diabetes analysis."""
        if not hasattr(self, 'merged_data') or self.merged_data.empty:
            print("❌ Error: No data available for training models.")
            return

        self.print_section_header("Model Training")
        
        # Define features for all models - only using real NHANES data
        print("\n🔄 Setting up features from NHANES dataset...")
        
        # Base features that we know exist
        self.common_features = [
            'glucose_level',
            'blood_pressure',
            'bmi',
            'age',
            'hdl',
            'triglycerides'
        ]

        # Only add additional features if they exist in the dataset
        if 'LBXGLT' in self.merged_data.columns:
            self.merged_data['glucose_tolerance'] = self.merged_data['LBXGLT']
            self.common_features.append('glucose_tolerance')
            print("  ✓ Added glucose tolerance from LBXGLT")
            
        if 'LBXGH' in self.merged_data.columns:
            self.merged_data['hba1c'] = self.merged_data['LBXGH']
            self.common_features.append('hba1c')
            print("  ✓ Added HbA1c from LBXGH")
            
        if 'LBXIN' in self.merged_data.columns:
            self.merged_data['insulin'] = self.merged_data['LBXIN']
            self.common_features.append('insulin')
            print("  ✓ Added insulin from LBXIN")
        elif 'LBDINSI' in self.merged_data.columns:
            self.merged_data['insulin'] = self.merged_data['LBDINSI']
            self.common_features.append('insulin')
            print("  ✓ Added insulin from LBDINSI")

        # Physical Activity (only if we have PAQ questions)
        paq_columns = [col for col in self.merged_data.columns if col.startswith('PAQ')]
        if paq_columns:
            # Convert selected PAQ columns to numeric, handling non-numeric values
            paq_data = self.merged_data[paq_columns].apply(pd.to_numeric, errors='coerce')
            # Normalize each column and take the mean for an activity score
            self.merged_data['physical_activity'] = paq_data.apply(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
            ).mean(axis=1)
            self.common_features.append('physical_activity')
            print("  ✓ Added physical activity score from PAQ questions")

        # Family History (only if DIQ170 exists)
        if 'DIQ170' in self.merged_data.columns:
            # DIQ170 is coded as 1 for yes, 2 for no
            self.merged_data['family_history'] = (self.merged_data['DIQ170'] == 1).astype(int)
            self.common_features.append('family_history')
            print("  ✓ Added family history from DIQ170")

        # Handle missing values in the real features
        print("\n🔧 Handling missing values...")
        for feature in self.common_features:
            missing = self.merged_data[feature].isna().sum()
            if missing > 0:
                print(f"  ⚠️  Removing {missing} rows with missing values in {feature}")
                self.merged_data = self.merged_data.dropna(subset=[feature])

        print(f"\n📊 Final feature set: {', '.join(self.common_features)}")
        print(f"📈 Number of complete cases: {len(self.merged_data)}")

        # Prepare features for modeling
        X = self.merged_data[self.common_features]

        # 1. Diabetes Classification Model (No Diabetes, Pre-diabetes, Type 1, Type 2)
        print("🔄 Training Diabetes Classification Model...")
        
        # Create diabetes classification labels
        def create_diabetes_labels(row):
            # Check if we have DIQ010 (diabetes diagnosis) from questionnaire
            if 'DIQ010' in self.merged_data.columns:
                if row['DIQ010'] == 1:  # Yes, diagnosed with diabetes
                    if row['age'] < 30:
                        return 2  # Type 1 (age-based approximation)
                    else:
                        return 3  # Type 2
                elif row['DIQ010'] == 3:  # Borderline
                    return 1  # Pre-diabetes
                elif row['DIQ010'] == 2:  # No
                    return 0  # No diabetes
            
            # If DIQ010 not available, use glucose levels
            if row['glucose_level'] < 100:
                return 0  # No Diabetes
            elif 100 <= row['glucose_level'] < 126:
                return 1  # Pre-diabetes
            else:
                if row['age'] < 30:
                    return 2  # Type 1 (age-based approximation)
                else:
                    return 3  # Type 2

        print("\n🔄 Creating diabetes classification labels...")
        y_diabetes = self.merged_data.apply(create_diabetes_labels, axis=1)
        
        # Ensure class balance
        if len(y_diabetes.unique()) < 4:
            print("  ⚠️  Adding synthetic samples to ensure class balance")
            from sklearn.utils import resample
            
            # Add synthetic samples for each missing class
            for class_label in range(4):
                if class_label not in y_diabetes.unique():
                    # Create synthetic samples
                    synthetic_size = len(y_diabetes) // 10
                    synthetic_X = pd.DataFrame(
                        np.random.normal(0, 1, (synthetic_size, len(self.common_features))),
                        columns=self.common_features
                    )
                    synthetic_y = pd.Series([class_label] * synthetic_size)
                    
                    # Add to training data
                    X = pd.concat([X, synthetic_X])
                    y_diabetes = pd.concat([y_diabetes, synthetic_y])
        
        X_train_diab, X_test_diab, y_train_diab, y_test_diab = train_test_split(
            X, y_diabetes, test_size=0.2, random_state=42
        )
        
        self.scaler_diabetes = StandardScaler()
        X_train_diab_scaled = self.scaler_diabetes.fit_transform(X_train_diab)
        X_test_diab_scaled = self.scaler_diabetes.transform(X_test_diab)
        
        self.diabetes_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
        self.diabetes_model.fit(X_train_diab_scaled, y_train_diab)
        
        # Print model performance
        train_score = self.diabetes_model.score(X_train_diab_scaled, y_train_diab)
        test_score = self.diabetes_model.score(X_test_diab_scaled, y_test_diab)
        print("\n📊 Diabetes Classification Model Performance:")
        print(f"  ├─ Training Accuracy: {train_score:.2%}")
        print(f"  └─ Testing Accuracy:  {test_score:.2%}")

        # 2. Risk Assessment Model
        print("\n🔄 Training Risk Assessment Model...")
        y_risk = (self.merged_data['glucose_level'] >= 126).astype(int)
        
        if len(y_risk.unique()) < 2:
            print("  ⚠️  Adding synthetic samples to ensure class balance")
            # Add some synthetic samples for the minority class
            minority_class = 1 if (y_risk == 1).sum() < (y_risk == 0).sum() else 0
            synthetic_samples = len(y_risk) // 10  # Add 10% synthetic samples
            
            synthetic_X = pd.DataFrame({
                'glucose_level': np.random.uniform(126 if minority_class == 1 else 70, 
                                                 200 if minority_class == 1 else 99, 
                                                 synthetic_samples),
                'blood_pressure': np.random.uniform(90, 180, synthetic_samples),
                'bmi': np.random.uniform(18.5, 40, synthetic_samples),
                'age': np.random.uniform(20, 80, synthetic_samples),
                'hdl': np.random.uniform(30, 90, synthetic_samples),
                'triglycerides': np.random.uniform(50, 200, synthetic_samples)
            })
            
            synthetic_y = pd.Series([minority_class] * synthetic_samples)
            
            X = pd.concat([X, synthetic_X])
            y_risk = pd.concat([y_risk, synthetic_y])
        
        X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
            X, y_risk, test_size=0.2, random_state=42
        )
        
        self.scaler_risk = StandardScaler()
        X_train_risk_scaled = self.scaler_risk.fit_transform(X_train_risk)
        X_test_risk_scaled = self.scaler_risk.transform(X_test_risk)
        
        self.risk_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.risk_model.fit(X_train_risk_scaled, y_train_risk)
        
        # Print model performance
        train_score = self.risk_model.score(X_train_risk_scaled, y_train_risk)
        test_score = self.risk_model.score(X_test_risk_scaled, y_test_risk)
        print("\n📊 Risk Model Performance:")
        print(f"  ├─ Training Accuracy: {train_score:.2%}")
        print(f"  └─ Testing Accuracy:  {test_score:.2%}")

        # 3. Complications Prediction Model
        # print("\n🔄 Training Complications Model...")
        
        # # Map NHANES variables to complications
        # complications_mapping = {
        #     'retinopathy': {
        #         'condition': 'DIQ070',  # Retinopathy diagnosis
        #         'value': 1  # Yes
        #     },
        #     'nephropathy': {
        #         'condition': 'KIQ022',  # Kidney problems
        #         'value': 1  # Yes
        #     },
        #     'neuropathy': {
        #         'condition': 'DIQ080',  # Numbness/tingling in hands/feet
        #         'value': 1  # Yes
        #     },
        #     'cardiovascular': {
        #         'condition': ['BPQ020', 'MCQ160B'],  # High blood pressure or heart disease
        #         'value': 1  # Yes
        #     }
        # }
        
        # # Create complications data from actual diagnoses
        # complications_data = pd.DataFrame()
        # available_complications = []
        
        # for complication, mapping in complications_mapping.items():
        #     conditions = mapping['condition'] if isinstance(mapping['condition'], list) else [mapping['condition']]
        #     has_data = False
            
        #     for condition in conditions:
        #         if condition in self.merged_data.columns:
        #             complications_data[complication] = (self.merged_data[condition] == mapping['value']).astype(int)
        #             available_complications.append(complication)
        #             has_data = True
        #             print(f"  ✓ Added {complication} data from {condition}")
        #             break
            
        #     if not has_data:
        #         print(f"  ⚠️  No data available for {complication}")
        
        # if not available_complications:
        #     print("❌ No complications data available - skipping complications model")
        #     return
            
        # # Train model only on available complications
        # X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
        #     X, complications_data[available_complications], test_size=0.2, random_state=42
        # )
        
        # # Initialize and fit the scaler for complications
        # self.scaler_comp = StandardScaler()
        # X_train_comp_scaled = self.scaler_comp.fit_transform(X_train_comp)
        # X_test_comp_scaled = self.scaler_comp.transform(X_test_comp)
        
        # # Train the complications model
        # self.complications_model = MultiOutputClassifier(
        #     GradientBoostingClassifier(
        #         n_estimators=100,
        #         max_depth=5,
        #         random_state=42
        #     )
        # )
        # self.complications_model.fit(X_train_comp_scaled, y_train_comp)
        
        # # Print model performance
        # train_score = self.complications_model.score(X_train_comp_scaled, y_train_comp)
        # test_score = self.complications_model.score(X_test_comp_scaled, y_test_comp)
        # print("\n📊 Complications Model Performance:")
        # print(f"  ├─ Training Accuracy: {train_score:.2%}")
        # print(f"  └─ Testing Accuracy:  {test_score:.2%}")

        # 4. Treatment Response Model
        print("\n🔄 Training Treatment Response Model...")
        
        # Map NHANES variables to treatments
        treatment_mapping = {
            'insulin': {
                'medications': ['INSULIN', 'INSULIN GLARGINE', 'INSULIN ASPART'],
                'conditions': ['E11', 'E10']  # Type 2 and Type 1 diabetes
            },
            'metformin': {
                'medications': ['METFORMIN', 'GLYBURIDE', 'GLIPIZIDE', 'SITAGLIPTIN'],
                'conditions': ['E11']  # Type 2 diabetes
            },
            'lifestyle_changes': {
                'medications': ['PHENTERMINE'],  # Weight loss medications
                'conditions': ['E66.3', 'E78.0', 'E78.1']  # Weight and metabolic conditions
            },
            'dietary_modification': {
                'conditions': ['E78.0', 'E78.1', 'E66.3']  # Cholesterol, triglycerides, overweight
            }
        }
        
        # Create treatment response data from actual responses
        treatment_response = pd.DataFrame()
        available_treatments = []
        
        for treatment, mapping in treatment_mapping.items():
            has_data = False
            
            # Check medications if specified
            if 'medications' in mapping:
                medications = mapping['medications']
                for med in medications:
                    if 'RXDDRUG' in self.medications_data.columns:
                        has_med = self.medications_data['RXDDRUG'].str.contains(med, case=False, na=False)
                        if has_med.any():
                            treatment_response[treatment] = has_med.astype(int)
                            available_treatments.append(treatment)
                            has_data = True
                            print(f"  ✓ Added {treatment} data from medication {med}")
                            break
            
            # Check conditions if specified
            if not has_data and 'conditions' in mapping:
                conditions = mapping['conditions']
                for condition in conditions:
                    if 'RXDRSD1' in self.medications_data.columns:
                        has_condition = self.medications_data['RXDRSD1'].str.contains(condition, case=False, na=False)
                        if has_condition.any():
                            treatment_response[treatment] = has_condition.astype(int)
                            available_treatments.append(treatment)
                            has_data = True
                            print(f"  ✓ Added {treatment} data from condition {condition}")
                            break
            
            if not has_data:
                print(f"  ⚠️  No data available for {treatment}")
        
        if not available_treatments:
            print("⚠️  No treatment response data available - generating synthetic data")
            # Generate synthetic treatment data
            synthetic_size = len(X) if 'X' in locals() else 1000
            available_treatments = ['insulin', 'metformin', 'lifestyle_changes', 'dietary_modification']
            
            # Create synthetic treatment responses based on patient characteristics
            treatment_response = pd.DataFrame()
            for treatment in available_treatments:
                if treatment == 'insulin':
                    # Higher probability for insulin if glucose level is high
                    prob = (self.merged_data['glucose_level'] > 180).astype(float) * 0.8  # Max 0.8 for high glucose
                    prob += (self.merged_data['glucose_level'] > 126).astype(float) * 0.4  # Add 0.4 for diabetic
                    prob += 0.2 * np.random.random(len(self.merged_data))  # Add some randomness
                    treatment_response[treatment] = (prob > 0.5).astype(int)
                elif treatment == 'metformin':
                    # Higher probability for metformin in type 2 cases
                    prob = (self.merged_data['glucose_level'] > 126).astype(float) * 0.6  # Base on glucose
                    prob += (self.merged_data['age'] > 40).astype(float) * 0.3  # Add age factor
                    prob += 0.1 * np.random.random(len(self.merged_data))  # Add some randomness
                    treatment_response[treatment] = (prob > 0.5).astype(int)
                elif treatment == 'lifestyle_changes':
                    # Higher probability for lifestyle changes if BMI is high
                    prob = (self.merged_data['bmi'] > 30).astype(float) * 0.7  # Obese
                    prob += (self.merged_data['bmi'] > 25).astype(float) * 0.4  # Overweight
                    prob += 0.2 * np.random.random(len(self.merged_data))  # Add some randomness
                    treatment_response[treatment] = (prob > 0.5).astype(int)
                else:  # dietary_modification
                    # Higher probability for dietary changes if glucose or BMI is high
                    prob = (self.merged_data['glucose_level'] > 126).astype(float) * 0.5  # High glucose
                    prob += (self.merged_data['bmi'] > 25).astype(float) * 0.4  # Overweight/obese
                    prob += (self.merged_data['triglycerides'] > 150).astype(float) * 0.3  # High triglycerides
                    prob += 0.1 * np.random.random(len(self.merged_data))  # Add some randomness
                    treatment_response[treatment] = (prob > 0.5).astype(int)
                print(f"  ✓ Generated synthetic data for {treatment}")
            
            print(f"  ✓ Created synthetic treatment responses for {len(available_treatments)} treatments")
            
        # Train model only on available treatments
        X_train_treat, X_test_treat, y_train_treat, y_test_treat = train_test_split(
            X, treatment_response[available_treatments], test_size=0.2, random_state=42
        )
        
        self.scaler_treat = StandardScaler()
        X_train_treat_scaled = self.scaler_treat.fit_transform(X_train_treat)
        X_test_treat_scaled = self.scaler_treat.transform(X_test_treat)
        
        self.treatment_model = MultiOutputClassifier(
            GradientBoostingClassifier(n_estimators=100, random_state=42)
        )
        self.treatment_model.fit(X_train_treat_scaled, y_train_treat)
        
        # Print treatment model performance
        treat_scores = [estimator.score(X_test_treat_scaled, y_test_treat[treatment])
                       for estimator, treatment in zip(self.treatment_model.estimators_, 
                                                     available_treatments)]
        print("\n📊 Treatment Model Performance:")
        for i, (treatment, score) in enumerate(zip(available_treatments, treat_scores)):
            prefix = "  └─ " if i == len(available_treatments) - 1 else "  ├─ "
            print(f"{prefix}{treatment.replace('_', ' ').title():<20} Accuracy: {score:.2%}")
        
        self.model_features = self.common_features
        self.treatment_options = available_treatments
        
        self.print_section_header("Training Complete")

    def determine_diabetes_type(self, patient_data: Dict) -> str:
        """Determine diabetes status and type using the trained classification model."""
        try:
            # Prepare features for prediction
            features = []
            for feature in self.common_features:
                features.append(float(patient_data.get(feature, 0)))
            
            # Scale features
            features_scaled = self.scaler_diabetes.transform(np.array(features).reshape(1, -1))
            
            # Get prediction
            diabetes_class = self.diabetes_model.predict(features_scaled)[0]
            
            # Get prediction probabilities
            class_probabilities = self.diabetes_model.predict_proba(features_scaled)[0]
            confidence = max(class_probabilities)
            
            # Map class to diagnosis
            diabetes_types = {
                0: 'No Diabetes',
                1: 'Pre-diabetes',
                2: 'Type 1',
                3: 'Type 2'
            }
            
            diagnosis = diabetes_types[diabetes_class]
            
            # If confidence is low, return Undetermined
            if confidence < 0.6:
                print(f"⚠️  Low confidence ({confidence:.1%}) in diabetes type prediction")
                return 'Undetermined'
                
            print(f"✓ Diabetes classification confidence: {confidence:.1%}")
            return diagnosis
            
        except Exception as e:
            print(f"❌ Error in diabetes type determination: {str(e)}")
            return 'Undetermined'

    def validate_features(self, features_dict: Dict) -> Tuple[bool, List[str]]:
        """Validate that required features are present and have valid values."""
        required_features = {
            'glucose_level': (0, 500),    # mg/dL
            'blood_pressure': (60, 250),  # mmHg
            'bmi': (10, 70),             # kg/m²
            'age': (0, 120),             # years
            'hdl': (0, 200),             # mg/dL
            'triglycerides': (0, 1000)   # mg/dL
        }
        
        missing_features = []
        invalid_features = []
        
        for feature, (min_val, max_val) in required_features.items():
            if feature not in features_dict:
                missing_features.append(feature)
            elif not isinstance(features_dict[feature], (int, float)):
                invalid_features.append(f"{feature} (invalid type)")
            elif not min_val <= float(features_dict[feature]) <= max_val:
                invalid_features.append(f"{feature} (out of range)")
        
        return len(missing_features + invalid_features) == 0, missing_features + invalid_features

    def prepare_features(self, patient_data: Dict) -> np.ndarray:
        """Prepare and validate features for prediction."""
        # Validate input data
        is_valid, issues = self.validate_features(patient_data)
        if not is_valid:
            raise ValueError(f"Invalid patient data: {', '.join(issues)}")

        # Prepare features in the correct order
        feature_values = []
        for feature in self.common_features:
            feature_values.append(float(patient_data[feature]))
        
        return np.array(feature_values).reshape(1, -1)

    def predict_diabetes_risk(self, patient_data: Dict) -> Tuple[float, Dict, Dict, Dict]:
        """
        Predict diabetes risk, complications, and recommended treatments.
        Returns (risk_probability, risk_factors, complications, treatments)
        """
        try:
            if not hasattr(self, 'risk_model'):
                self.print_section_header("Model Initialization")
                print("🔄 Training models for first use...")
                self.train_models()

            # Prepare and validate features
            try:
                patient_features = self.prepare_features(patient_data)
            except ValueError as e:
                print(f"\n❌ Error: {str(e)}")
                print("⚠️  Using default values for missing/invalid features:")
                default_values = {
                    'glucose_level': 100,
                    'blood_pressure': 120,
                    'bmi': 25,
                    'age': 45,
                    'hdl': 50,
                    'triglycerides': 150
                }
                for feature in self.common_features:
                    if feature not in patient_data or not isinstance(patient_data[feature], (int, float)):
                        print(f"  ├─ {feature}: {default_values[feature]} (default)")
                        patient_data[feature] = default_values[feature]
                patient_features = self.prepare_features(patient_data)

            # Scale features
            risk_features_scaled = self.scaler_risk.transform(patient_features)
            # comp_features_scaled = self.scaler_comp.transform(patient_features)
            treat_features_scaled = self.scaler_treat.transform(patient_features)

            # Get predictions
            risk_prob = self.risk_model.predict_proba(risk_features_scaled)[0][1]
            # complications_prob = self.predict_complications(comp_features_scaled)
            treatment_effectiveness = self.recommend_treatments(treat_features_scaled)

            # Calculate feature importances
            feature_importance = dict(zip(self.model_features, 
                                        self.risk_model.feature_importances_))

            # Prepare risk factors analysis with validation
            risk_factors = {}
            for feature in ['glucose_level', 'blood_pressure', 'bmi']:
                value = patient_data.get(feature, 0)
                
                # Validate value ranges
                if feature == 'glucose_level':
                    status = ('High' if value > 125 else 
                             'Elevated' if value > 100 else 'Normal')
                elif feature == 'blood_pressure':
                    status = ('High' if value > 130 else 
                             'Elevated' if value > 120 else 'Normal')
                elif feature == 'bmi':
                    status = ('Obese' if value > 30 else 
                             'Overweight' if value > 25 else 'Normal')
                
                risk_factors[feature] = {
                    'value': value,
                    'importance': feature_importance.get(feature, 0),
                    'status': status
                }

            return risk_prob, risk_factors, treatment_effectiveness

        except Exception as e:
            print(f"\n❌ Error during prediction: {str(e)}")
            print("⚠️  Please ensure all required features are provided with valid values:")
            print("  Required features and valid ranges:")
            print("  ├─ glucose_level: 0-500 mg/dL")
            print("  ├─ blood_pressure: 60-250 mmHg")
            print("  ├─ bmi: 10-70 kg/m²")
            print("  ├─ age: 0-120 years")
            print("  ├─ hdl: 0-200 mg/dL")
            print("  └─ triglycerides: 0-1000 mg/dL")
            raise

    # def predict_complications(self, patient_features_scaled: np.ndarray) -> Dict[str, float]:
    #     """Predict probability of various complications."""
    #     try:
    #         complication_probs = self.complications_model.predict_proba(patient_features_scaled)
    #         return {
    #             complication: probs[0][1]
    #             for complication, probs in zip(self.complications, complication_probs)
    #         }
    #     except Exception as e:
    #         print(f"\n❌ Error predicting complications: {str(e)}")
    #         return {complication: 0.0 for complication in self.complications}

    def recommend_treatments(self, patient_features_scaled: np.ndarray) -> Dict[str, float]:
        """Predict effectiveness of different treatments."""
        try:
            # Get the raw features back from scaled features
            patient_features = self.scaler_treat.inverse_transform(patient_features_scaled)[0]
            feature_dict = dict(zip(self.common_features, patient_features))
            
            # Calculate treatment confidences based on clinical thresholds
            treatment_effectiveness = {}
            
            # Insulin confidence
            if feature_dict['glucose_level'] > 180:
                insulin_conf = 0.9  # Very high confidence for severe hyperglycemia
            elif feature_dict['glucose_level'] > 126:
                insulin_conf = 0.6  # Moderate confidence for diabetes range
            elif feature_dict['glucose_level'] > 100:
                insulin_conf = 0.3  # Low confidence for pre-diabetes
            else:
                insulin_conf = 0.1  # Very low confidence for normal glucose
            treatment_effectiveness['insulin'] = insulin_conf
            
            # Metformin confidence
            if feature_dict['glucose_level'] > 126 and feature_dict['age'] > 40:
                metformin_conf = 0.8  # High confidence for type 2 diabetes profile
            elif feature_dict['glucose_level'] > 100:
                metformin_conf = 0.5  # Moderate confidence for pre-diabetes
            else:
                metformin_conf = 0.2  # Low confidence for normal glucose
            treatment_effectiveness['metformin'] = metformin_conf
            
            # Lifestyle changes confidence
            lifestyle_conf = 0.0
            if feature_dict['bmi'] > 30:
                lifestyle_conf += 0.4  # High impact for obesity
            elif feature_dict['bmi'] > 25:
                lifestyle_conf += 0.3  # Moderate impact for overweight
            if feature_dict['glucose_level'] > 100:
                lifestyle_conf += 0.3  # Additional impact for elevated glucose
            if feature_dict['blood_pressure'] > 130:
                lifestyle_conf += 0.2  # Additional impact for high blood pressure
            treatment_effectiveness['lifestyle_changes'] = min(lifestyle_conf, 0.9)  # Cap at 0.9
            
            # Dietary modification confidence
            diet_conf = 0.0
            if feature_dict['glucose_level'] > 126:
                diet_conf += 0.3  # Impact for diabetes range
            elif feature_dict['glucose_level'] > 100:
                diet_conf += 0.2  # Impact for pre-diabetes
            if feature_dict['bmi'] > 25:
                diet_conf += 0.2  # Impact for overweight
            if feature_dict['triglycerides'] > 150:
                diet_conf += 0.2  # Impact for high triglycerides
            if feature_dict['hdl'] < 40:
                diet_conf += 0.2  # Impact for low HDL
            treatment_effectiveness['dietary_modification'] = min(diet_conf, 0.9)  # Cap at 0.9
            
            return treatment_effectiveness
            
        except Exception as e:
            print(f"\n❌ Error recommending treatments: {str(e)}")
            return {treatment: 0.0 for treatment in self.treatment_options}

    def generate_treatment_plan(self, patient_data: Dict) -> Dict:
        """Generate personalized treatment plan using ML models."""
        self.print_section_header("Treatment Plan")
        
        # Get predictions from all models
        risk_prob, risk_factors, treatment_effectiveness = self.predict_diabetes_risk(patient_data)
        diabetes_type = self.determine_diabetes_type(patient_data)
        
        # Create treatment plan dictionary
        treatment_plan = {
            'diabetes_type': diabetes_type,
            'risk_level': 'High' if risk_prob > 0.7 else 'Medium' if risk_prob > 0.3 else 'Low',
            'risk_probability': risk_prob,
            'risk_factors': risk_factors,
            'recommended_treatments': treatment_effectiveness,
            'recommendations': []
        }
        
        # Format and display the treatment plan
        print(f"🏥 Patient Profile:")
        print(f"  ├─ Diabetes Type: {diabetes_type}")
        print(f"  ├─ Risk Level: {risk_prob:.1%}")
        print(f"  └─ Current Status: {treatment_plan['risk_level']} Risk")
        
        print("\n📊 Risk Factors:")
        for factor, data in risk_factors.items():
            status_color = "🔴" if data['status'] in ['High', 'Obese'] else "🟡" if data['status'] in ['Elevated', 'Overweight'] else "🟢"
            print(f"  ├─ {factor.title():<15} {status_color} {data['status']}")
            # Add to recommendations based on status
            if data['status'] in ['High', 'Obese', 'Elevated', 'Overweight']:
                if factor == 'glucose_level':
                    treatment_plan['recommendations'].append(
                        f"Focus on glucose management - Current level is {data['status']}"
                    )
                elif factor == 'blood_pressure':
                    treatment_plan['recommendations'].append(
                        f"Monitor blood pressure - Current level is {data['status']}"
                    )
                elif factor == 'bmi':
                    treatment_plan['recommendations'].append(
                        f"Weight management program recommended - BMI status: {data['status']}"
                    )
        
        # print("\n⚠️ Complications Risk:")
        # for complication, risk in complications_prob.items():
        #     risk_level = "High" if risk > 0.7 else "Medium" if risk > 0.3 else "Low"
        #     risk_color = "🔴" if risk > 0.7 else "🟡" if risk > 0.3 else "🟢"
        #     print(f"  ├─ {complication.title():<15} {risk_color} {risk_level} ({risk:.1%})")
        
        print("\n💊 Recommended Treatments:")
        sorted_treatments = sorted(treatment_effectiveness.items(), key=lambda x: x[1], reverse=True)
        for treatment, effectiveness in sorted_treatments:
            if effectiveness > 0.5:
                print(f"  ├─ {treatment.replace('_', ' ').title():<20} Effectiveness: {effectiveness:.1%}")
                treatment_plan['recommendations'].append(
                    f"{treatment.replace('_', ' ').title()}: {effectiveness:.1%} expected effectiveness"
                )
        
        print("\n📋 Action Plan:")
        # Add type-specific recommendations
        if diabetes_type == 'Type 1':
            type_specific_recs = [
                "Regular insulin therapy required",
                "Frequent blood glucose monitoring (4-8 times daily)",
                "Carbohydrate counting for meal planning"
            ]
            treatment_plan['recommendations'].extend(type_specific_recs)
            for rec in type_specific_recs:
                print(f"  ├─ {rec}")
        else:  # Type 2 or Undetermined
            type_specific_recs = [
                "Regular blood glucose monitoring",
                "Balanced diet with controlled carbohydrate intake",
                "Regular physical activity (at least 150 minutes per week)"
            ]
            treatment_plan['recommendations'].extend(type_specific_recs)
            for rec in type_specific_recs:
                print(f"  ├─ {rec}")

        # Add risk-level specific recommendations
        if risk_prob >= 0.7:
            high_risk_recs = [
                "Immediate consultation with endocrinologist",
                "Daily blood glucose monitoring",
                "Strict dietary control",
                "Consider medication adjustment"
            ]
            treatment_plan['recommendations'].extend(high_risk_recs)
            for rec in high_risk_recs:
                print(f"  ├─ {rec}")
        elif risk_prob >= 0.3:
            medium_risk_recs = [
                "Regular check-ups with healthcare provider",
                "Weekly blood glucose monitoring",
                "Moderate dietary restrictions"
            ]
            treatment_plan['recommendations'].extend(medium_risk_recs)
            for rec in medium_risk_recs:
                print(f"  ├─ {rec}")

        # Add complication-specific recommendations
        # for complication, prob in complications_prob.items():
        #     if prob > 0.5:
        #         complication_recs = []
        #         if complication == 'retinopathy':
        #             complication_recs = [
        #                 "Regular eye examinations",
        #                 "Strict blood sugar control to prevent vision problems",
        #                 "Consider ophthalmologist consultation"
        #             ]
        #         elif complication == 'nephropathy':
        #             complication_recs = [
        #                 "Regular kidney function tests",
        #                 "Blood pressure management",
        #                 "Protein intake monitoring"
        #             ]
        #         elif complication == 'neuropathy':
        #             complication_recs = [
        #                 "Regular foot examinations",
        #                 "Proper foot care and hygiene",
        #                 "Pain management if needed"
        #             ]
        #         elif complication == 'cardiovascular':
        #             complication_recs = [
        #                 "Regular cardiovascular check-ups",
        #                 "Cholesterol management",
        #                 "Blood pressure monitoring"
        #             ]
                
        #         treatment_plan['recommendations'].extend(complication_recs)
        #         for rec in complication_recs:
        #             print(f"  ├─ {rec}")

        # Add a summary section to the treatment plan
        treatment_plan['summary'] = {
            'total_recommendations': len(treatment_plan['recommendations']),
            'key_areas_of_concern': [
                factor for factor, data in risk_factors.items()
                if data['status'] in ['High', 'Obese', 'Elevated', 'Overweight']
            ],
            # 'high_risk_complications': [
            #     complication for complication, risk in complications_prob.items()
            #     if risk > 0.7
            # ],
            'primary_treatments': [
                treatment for treatment, effectiveness in sorted_treatments
                if effectiveness > 0.7
            ]
        }

        print("\n📝 Summary:")
        print(f"  ├─ Total Recommendations: {treatment_plan['summary']['total_recommendations']}")
        if treatment_plan['summary']['key_areas_of_concern']:
            print(f"  ├─ Key Areas of Concern: {', '.join(treatment_plan['summary']['key_areas_of_concern'])}")
        # if treatment_plan['summary']['high_risk_complications']:
        #     print(f"  ├─ High Risk Complications: {', '.join(treatment_plan['summary']['high_risk_complications'])}")
        if treatment_plan['summary']['primary_treatments']:
            print(f"  └─ Primary Treatments: {', '.join(treatment_plan['summary']['primary_treatments'])}")
        
        return treatment_plan
    
    def visualize_health_metrics(self, patient_data: Dict):
        """Create visualizations of key health metrics with reference ranges and risk indicators."""
        if not hasattr(self, 'merged_data') or self.merged_data.empty:
            print("Warning: No population data available for comparison.")
            return
            
        # Use a clean, modern style
        plt.style.use('default')
        fig = plt.figure(figsize=(15, 10))
        
        # Define reference ranges and risk levels
        reference_ranges = {
            'glucose_level': {
                'normal': (70, 99),
                'pre_diabetic': (100, 125),
                'diabetic': (126, 200)
            },
            'blood_pressure': {
                'normal': (90, 120),
                'elevated': (121, 130),
                'high': (131, 180)
            },
            'bmi': {
                'underweight': (0, 18.5),
                'normal': (18.5, 24.9),
                'overweight': (25, 29.9),
                'obese': (30, 40)
            }
        }
        
        # Create multiple subplots for different metrics
        metrics = ['glucose_level', 'blood_pressure', 'bmi', 'age']
        for i, metric in enumerate(metrics, 1):
            if metric in patient_data:
                ax = plt.subplot(2, 2, i)
                
                # Plot patient value
                value = patient_data[metric]
                bar = plt.bar(['Patient'], [value], color='#4285F4', alpha=0.7, width=0.4)
                
                # Add value label on top of the bar
                plt.text(0, value, f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
                
                # Set title and labels
                plt.title(f'Patient {metric.replace("_", " ").title()}', pad=20, fontsize=12, fontweight='bold')
                
                # Add reference ranges if available
                if metric in reference_ranges:
                    ranges = reference_ranges[metric]
                    colors = ['#34A853', '#FBBC05', '#EA4335']  # Google-style colors
                    alpha = 0.15
                    
                    # Plot reference ranges as background
                    for (level, (min_val, max_val)), color in zip(ranges.items(), colors):
                        plt.axhspan(min_val, max_val, color=color, alpha=alpha, 
                                  label=f'{level.replace("_", " ").title()}: {min_val}-{max_val}')
                    
                    # Determine patient's range and add status text
                    for level, (min_val, max_val) in ranges.items():
                        if min_val <= value <= max_val:
                            status_text = f'Status:\n{level.replace("_", " ").title()}'
                            plt.text(0, value/2, status_text,
                                   ha='center', va='center', fontweight='bold',
                                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
                    
                    # Adjust legend position and style
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                             frameon=True, fancybox=True, shadow=True)
                
                # Add population average if available
                if metric in self.merged_data.columns:
                    avg = self.merged_data[metric].mean()
                    plt.axhline(y=avg, color='#EA4335', linestyle='--', linewidth=2,
                              label=f'Population Average: {avg:.1f}')
                    
                # Customize y-axis
                if metric == 'glucose_level':
                    plt.ylim(0, max(200, value * 1.2))
                elif metric == 'blood_pressure':
                    plt.ylim(0, max(180, value * 1.2))
                elif metric == 'bmi':
                    plt.ylim(0, max(40, value * 1.2))
                else:
                    plt.ylim(0, value * 1.2)
                
                # Remove x-axis labels and add grid
                plt.xticks([])
                plt.grid(True, alpha=0.2, linestyle='--')
                
                # Add subtle box around the plot
                for spine in ax.spines.values():
                    spine.set_color('#CCCCCC')
                    spine.set_linewidth(0.5)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

def main():
    try:
        # Initialize the treatment planner
        planner = DiabetesTreatmentPlanner()
        planner.preprocess_data()
        
        # Train all models
        planner.train_models()
        
        # Example patient data
        example_patient = {
            'age': 45,
            'bmi': 28.5,
            'blood_pressure': 135,
            'glucose_level': 110,
            'hdl_cholesterol': 45,
            'triglycerides': 160,
            'glucose_tolerance': 145,
            'insulin': 15,
            'family_history': 1,
            'physical_activity': 0.7,
            'healthy_diet': 0.6,
            'autoimmune_conditions': False
        }
        
        # Generate and display treatment plan
        treatment_plan = planner.generate_treatment_plan(example_patient)
        
        print("\nPersonalized Treatment Plan:")
        print(f"Diabetes Type: {treatment_plan['diabetes_type']}")
        print(f"Risk Level: {treatment_plan['risk_level']}")
        print(f"Risk Probability: {treatment_plan['risk_probability']}")
        
        print("\nRisk Factors Analysis:")
        for factor, data in treatment_plan['risk_factors'].items():
            print(f"{factor.title()}: {data['status']} (Importance: {data['importance']:.2f})")
        
        print("\nComplications Risk:")
        for complication, risk in treatment_plan['complications_risk'].items():
            print(f"{complication.title()}: {risk:.1%} risk")
        
        print("\nRecommendations:")
        for i, rec in enumerate(treatment_plan['recommendations'], 1):
            print(f"{i}. {rec}")
        
        # Visualize patient's health metrics
        planner.visualize_health_metrics(example_patient)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check if all data files are present and properly formatted.")

if __name__ == "__main__":
    main()
