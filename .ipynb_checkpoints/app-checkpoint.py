import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification")
st.write("Predict whether an employee earns >50K or <=50K based on their profile.")

# -----------------------
# Sidebar: Single Input
# -----------------------
st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 18, 70, 30)
education = st.sidebar.selectbox("Education Level", [
    'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
    'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', '5th-6th'
])
occupation = st.sidebar.selectbox("Occupation", [
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
    'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
    'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
experience = st.sidebar.slider("Years of Experience", 0, 50, 5)

# -----------------------
# Create input DataFrame
# -----------------------
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours_per_week': [hours_per_week],
    'experience': [experience]
})

# -----------------------
# Predict Button
# -----------------------
if st.sidebar.button("Predict Salary"):
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction Result:")
    st.success(f"The employee is likely to earn **{prediction}**")

# -----------------------
# Batch Prediction
# -----------------------
st.markdown("---")
st.subheader("📁 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with employee details", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    try:
        predictions = model.predict(data)
        data['Predicted Salary'] = predictions
        st.write("✅ Predictions completed:")
        st.dataframe(data)

        # Download CSV
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results CSV", data=csv, file_name="predicted_salaries.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error in prediction: {str(e)}")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown(
    "<small>Built with ❤️ using Streamlit | © 2025</small>",
    unsafe_allow_html=True
)
