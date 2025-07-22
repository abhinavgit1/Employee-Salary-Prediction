import streamlit as st
import pandas as pd
import joblib

# --- 1. Load Model ---
# Load the pre-trained model. Ensure 'best_model.pkl' is in the same directory.
try:
    model = joblib.load("best_model.pkl")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- 2. Hardcoded Information ---
# This information is derived from your notebook and dataset.

# Define the order of features the model expects
FEATURE_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 
    'occupation', 'relationship', 'race', 'gender', 'capital-gain', 
    'capital-loss', 'hours-per-week', 'native-country'
]

# Options for the dropdown menus in the sidebar
CATEGORICAL_OPTIONS = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Local-gov', 'Others', 'State-gov', 'Self-emp-inc', 'Federal-gov'],
    'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Widowed', 'Separated', 'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation': ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv', 'Others', 'Other-service', 'Prof-specialty', 'Craft-repair', 'Adm-clerical', 'Exec-managerial', 'Tech-support', 'Sales', 'Priv-house-serv', 'Transport-moving', 'Handlers-cleaners', 'Armed-Forces'],
    'relationship': ['Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife', 'Other-relative'],
    'race': ['Black', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    'gender': ['Male', 'Female'],
    'native-country': ['United-States', 'Cuba', 'Jamaica', 'India', 'Others', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands']
}

# Mappings to convert categorical inputs to numbers for the model
LABEL_MAPPINGS = {
    'workclass': {'Federal-gov': 0, 'Local-gov': 1, 'Others': 2, 'Private': 3, 'Self-emp-inc': 4, 'Self-emp-not-inc': 5, 'State-gov': 6},
    'marital-status': {'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2, 'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6},
    'occupation': {'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 'Exec-managerial': 3, 'Farming-fishing': 4, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Other-service': 7, 'Others': 8, 'Priv-house-serv': 9, 'Prof-specialty': 10, 'Protective-serv': 11, 'Sales': 12, 'Tech-support': 13, 'Transport-moving': 14},
    'relationship': {'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3, 'Unmarried': 4, 'Wife': 5},
    'race': {'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4},
    'gender': {'Female': 0, 'Male': 1},
    'native-country': {'Cambodia': 0, 'Canada': 1, 'China': 2, 'Columbia': 3, 'Cuba': 4, 'Dominican-Republic': 5, 'Ecuador': 6, 'El-Salvador': 7, 'England': 8, 'France': 9, 'Germany': 10, 'Greece': 11, 'Guatemala': 12, 'Haiti': 13, 'Holand-Netherlands': 14, 'Honduras': 15, 'Hong': 16, 'Hungary': 17, 'India': 18, 'Iran': 19, 'Ireland': 20, 'Italy': 21, 'Jamaica': 22, 'Japan': 23, 'Laos': 24, 'Mexico': 25, 'Nicaragua': 26, 'Others': 27, 'Outlying-US(Guam-USVI-etc)': 28, 'Peru': 29, 'Philippines': 30, 'Poland': 31, 'Portugal': 32, 'Puerto-Rico': 33, 'Scotland': 34, 'South': 35, 'Taiwan': 36, 'Thailand': 37, 'Trinadad&Tobago': 38, 'United-States': 39, 'Vietnam': 40, 'Yugoslavia': 41}
}

# --- 3. UI Configuration ---
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="wide")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on their details.")

# --- 4. Sidebar for Single Prediction ---
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 17, 90, 38)
workclass = st.sidebar.selectbox("Work Class", CATEGORICAL_OPTIONS['workclass'])
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 12285, 1490400, 189188)
educational_num = st.sidebar.slider("Years of Education", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", CATEGORICAL_OPTIONS['marital-status'])
occupation = st.sidebar.selectbox("Occupation", CATEGORICAL_OPTIONS['occupation'])
relationship = st.sidebar.selectbox("Relationship", CATEGORICAL_OPTIONS['relationship'])
race = st.sidebar.selectbox("Race", CATEGORICAL_OPTIONS['race'])
gender = st.sidebar.selectbox("Gender", CATEGORICAL_OPTIONS['gender'])
capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 4356, 0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", CATEGORICAL_OPTIONS['native-country'])

# --- 5. Single Prediction Logic ---
# Create a DataFrame for the single input
input_dict = {
    'age': [age], 'workclass': [workclass], 'fnlwgt': [fnlwgt],
    'educational-num': [educational_num], 'marital-status': [marital_status],
    'occupation': [occupation], 'relationship': [relationship], 'race': [race],
    'gender': [gender], 'capital-gain': [capital_gain], 'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week], 'native-country': [native_country]
}
input_df = pd.DataFrame(input_dict)

st.write("### 🔎 Input Data for Single Prediction")
st.write(input_df)

# Preprocess the single input DataFrame
processed_df = input_df.copy()
for col, mapping in LABEL_MAPPINGS.items():
    processed_df[col] = processed_df[col].map(mapping)
processed_df = processed_df[FEATURE_COLUMNS]

# Predict button for single prediction
if st.button("Predict Salary Class"):
    prediction = model.predict(processed_df)
    st.success(f"✅ Prediction: The employee's salary is likely to be **{prediction[0]}**")

# --- 6. Batch Prediction Logic ---
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:", batch_data.head())
        
        # Keep a copy of the original data to show the user
        original_batch_data = batch_data.copy()
        
        # Preprocess the uploaded data
        processed_batch = batch_data.copy()
        for col, mapping in LABEL_MAPPINGS.items():
            if col in processed_batch.columns:
                processed_batch[col] = processed_batch[col].map(mapping)
        
        # **FIX:** Check for and handle NaNs that result from unseen categories
        if processed_batch.isnull().sum().any():
            st.warning("⚠️ Your file contained unrecognized categorical values. These rows have been handled automatically by filling with a default value (0), but predictions for them may be less accurate.")
            # Fill NaNs with 0 (a safe default corresponding to an encoded category)
            processed_batch.fillna(0, inplace=True)

        # Ensure all required columns are present and in order
        processed_batch = processed_batch[FEATURE_COLUMNS]
        
        # Make predictions
        batch_preds = model.predict(processed_batch)
        original_batch_data['Predicted_Income'] = batch_preds
        
        st.write("✅ Predictions:")
        st.write(original_batch_data.head())
        
        # Create a CSV for download
        csv = original_batch_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Predictions CSV", 
            csv, 
            file_name='predicted_salaries.csv', 
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"An error occurred during batch prediction: {e}")
        st.info("Please ensure your CSV file has the correct columns and format.")

