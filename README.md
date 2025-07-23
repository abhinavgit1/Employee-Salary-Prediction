<img width="1670" height="123" alt="image" src="https://github.com/user-attachments/assets/11d19738-41dc-4b41-8efb-6ab5e36d1655" /># Employee Salary Prediction Web App

![IBM SkillBuild](https://img.shields.io/badge/IBM%20SkillBuild-Internship%20Project-blue)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)](https://employee-salary-prediction-nvemncdd2i3cevace2l4zb.streamlit.app/)

This project is a machine learning web application that predicts whether an individual's annual income is likely to be more or less than $50,000. The model is trained on the US Adult Census dataset and deployed as an interactive web app using Streamlit.

This project was developed as part of the **Edunet Foundation's IBM SkillBuild Internship Program**.

---

## Live Demo

You can access the live application here:
**[https://your-streamlit-app-url.streamlit.app/](https://employee-salary-prediction-nvemncdd2i3cevace2l4zb.streamlit.app/)** *(Note: Replace this with your actual deployed Streamlit URL)*

---

## Problem Statement

The goal of this project is to build a highly accurate classification model to predict income levels based on a set of demographic and employment-related features. This involves handling a dataset with a mix of categorical and continuous variables, performing necessary data cleaning and preprocessing, and selecting the best-performing model for deployment. The final output is a user-friendly web interface where users can get instant predictions.

---

## Features

* **Single Prediction:** A sidebar form to input individual employee details and get an immediate salary class prediction.
* **Batch Prediction:** Upload a CSV file with multiple employee records to get predictions for all of them at once.
* **Downloadable Results:** Download the batch predictions as a new CSV file.
* **Responsive UI:** The application is designed to be accessible on both desktop and mobile devices.

---

## Tech Stack

* **Programming Language:** Python 3.12
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Model Saving/Loading:** Joblib
* **Web Framework:** Streamlit
* **Development Environment:** Jupyter Notebook, VS Code

---

## Project Structure

```
.
├── .streamlit/
│   └── config.toml      # Specifies the Python version for deployment
├── best_model.pkl       # The saved, trained Gradient Boosting model
├── salary_app.py        # The main Streamlit application script
├── requirements.txt     # List of Python dependencies for deployment
└── README.md            # This file
```

---

## Setup and Installation

To run this project on your local machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/employee-salary-prediction.git](https://github.com/abhinavgit1/Employee-Salary-Prediction.git)<img width="1670" height="123" alt="image" src="https://github.com/user-attachments/assets/b0b6cb83-5ee0-42ed-a7eb-371ed16a03fd" />
)
cd employee-salary-prediction
```

**2. Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

---

## Usage

Once the setup is complete, you can run the Streamlit application with the following command:

```bash
streamlit run salary_app.py
```

Navigate to `http://localhost:8501` in your web browser to view and interact with the application.

---

## Model Training and Selection

The machine learning model was developed in the `employee salary prediction.ipynb` notebook. The key steps were:

1.  **Data Loading & Cleaning:** The dataset was loaded, and missing values (represented by '?') were cleaned and handled.
2.  **Preprocessing:** Categorical features were converted to numerical format using `LabelEncoder`, and outliers were managed.
3.  **Model Comparison:** Several classification algorithms were trained and evaluated, including:
    * Logistic Regression
    * Random Forest
    * K-Nearest Neighbors (KNN)
    * Support Vector Machine (SVM)
    * **Gradient Boosting Classifier**
4.  **Final Model:** The **Gradient Boosting Classifier** was selected as the best-performing model, achieving an accuracy of **85.7%** on the test set. This model was then saved as `best_model.pkl`.

---
