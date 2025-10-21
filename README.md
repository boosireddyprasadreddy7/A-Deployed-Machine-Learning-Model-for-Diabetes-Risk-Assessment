# A-Deployed-Machine-Learning-Model-for-Diabetes-Risk-Assessment

## Overview

This project focuses on developing and deploying an end-to-end machine learning application to assess the risk of diabetes based on key health indicators. The project covers the complete machine learning lifecycle, including:

1.  **Exploratory Data Analysis (EDA):** Understanding the data patterns and distributions.
2.  **Feature Engineering (FE):** Creating new meaningful features from existing ones.
3.  **Data Preprocessing:** Handling outliers and encoding categorical features.
4.  **Model Training:** Evaluating multiple classification algorithms (Logistic Regression, KNN, Decision Tree, Random Forest, SVC).
5.  **Hyperparameter Tuning:** Optimizing the selected models using GridSearchCV.
6.  **Model Selection:** Identifying the best-performing model (KNN in this case).
7.  **Deployment:** Creating a user-friendly web application using Streamlit to interact with the trained model.

The goal is to predict whether a patient has diabetes (1) or not (0) based on diagnostic measurements.

## Dataset

The dataset used is originally from the **National Institute of Diabetes and Digestive and Kidney Diseases**. All patients are females at least 21 years old of Pima Indian heritage.

The original dataset consists of the following features:

* **Pregnancies**: Number of times pregnant
* **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
* **BloodPressure**: Diastolic blood pressure (mm Hg)
* **SkinThickness**: Triceps skin fold thickness (mm)
* **Insulin**: 2-Hour serum insulin (mu U/ml)
* **BMI**: Body mass index (weight in kg / (height in m)^2)
* **DiabetesPedigreeFunction**: Diabetes pedigree function
* **Age**: Age (years)
* **Outcome**: Class variable (0 or 1) - Target Variable

*Note: In the feature engineering step, `Glucose` and `BMI` were converted into categorical features (`NewGlucose`, `NewBMI`), and then one-hot encoded for model training. The original `Glucose` and `BMI` columns were dropped before training.*

## Features Used for Prediction in the Web App

The deployed web application requires the following inputs:

* **Pregnancies**: Number of times pregnant (numeric)
* **Blood Pressure**: Diastolic blood pressure (numeric)
* **Skin Thickness**: Triceps skin fold thickness (numeric)
* **Insulin**: 2-Hour serum insulin (numeric)
* **Diabetes Pedigree Function**: Diabetes pedigree function value (numeric)
* **Age**: Age in years (numeric)
* **BMI Category**: Select one (Obesity 1, Obesity 2, Obesity 3, Overweight, Underweight, Normal - *Note: 'Normal' category seems missing in app.py, encoded implicitly*)
* **Glucose Category**: Select one (Normal Glucose, Overweight Glucose, Secret Glucose - *Note: 'Low' and 'High' seem missing in app.py, encoded implicitly*)

## Project Structure



## Workflow

1.  **Data Loading & EDA:** Loaded the `diabetes.csv` dataset and performed initial analysis (shape, info, describe, correlations, outcome distribution). (See `01_EDA_AND_FE.ipynb`)
2.  **Data Preprocessing:**
    * Checked for missing values (none found).
    * Analyzed and handled outliers using IQR capping for relevant features.
    * Performed Feature Engineering: Created categorical `NewBMI` and `NewGlucose` features based on domain knowledge/thresholds.
    * Applied One-Hot Encoding to the newly created categorical features (`NewBMI`, `NewGlucose`).
    * Dropped original `Glucose`, `BMI`, and intermediate `NewInsulinScore` columns.
    * Saved the processed data to `diabetes_processed_data.csv`. (See `01_EDA_AND_FE.ipynb`)
3.  **Model Training & Evaluation:**
    * Split the processed data into training and testing sets.
    * Applied `StandardScaler` to the features.
    * Trained and evaluated baseline models: Logistic Regression, KNN, Decision Tree, Random Forest, SVC.
    * Performed Hyperparameter Tuning using `GridSearchCV` for each model type.
    * Selected K-Nearest Neighbors (KNN) as the best model based on test accuracy after tuning.
    * Saved the fitted `StandardScaler` object (`scaler.pkl`) and the best KNN model (`best_model.pkl`). (See `02_Model_Training.ipynb`)
4.  **Deployment:**
    * Created a Streamlit web application (`app.py`) that loads the scaler and the best model.
    * The app provides a user interface to input patient details and get a diabetes prediction. (See `app.py`)

## Technology Stack

* Python
* [cite_start]Pandas [cite: 1]
* [cite_start]NumPy [cite: 1]
* [cite_start]Scikit-learn [cite: 1]
* [cite_start]Matplotlib [cite: 1]
* [cite_start]Seaborn [cite: 1]
* [cite_start]Streamlit [cite: 1]
* Pickle (for model saving/loading)
* Jupyter Notebooks (for analysis and training)

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/boosireddyprasaddreddy7/A-Deployed-Machine-Learning-Model-for-Diabetes-Risk-Assessment.git](https://github.com/boosireddyprasaddreddy7/A-Deployed-Machine-Learning-Model-for-Diabetes-Risk-Assessment.git)
    cd A-Deployed-Machine-Learning-Model-for-Diabetes-Risk-Assessment
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
5.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`). Enter the patient details in the web interface and click "Predict".





