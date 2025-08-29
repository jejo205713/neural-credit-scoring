# neural-credit-scoring
# Intelligent Credit Risk Analysis Platform

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2+-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.1+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, web-based dashboard for training, evaluating, and utilizing a machine learning model for credit risk assessment. This platform provides an interactive and analytical interface to move beyond static scripts and notebooks into a dynamic, production-ready environment.

## 📸 Application Preview

Here is a preview of the "Predict User" page, showcasing the clean UI, interactive data visualization, and clear prediction output.

<table>
  <tr>
    <td align="center"><strong>Predict Single User</strong></td>
    <td align="center"><strong>Predict All Users</strong></td>
    <td align="center"><strong>Add New User</strong></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/jejo205713/neural-credit-scoring/raw/main/Images/Predict-single-user.png" alt="Predict Single User Page" />
    </td>
    <td>
      <img src="https://github.com/jejo205713/neural-credit-scoring/raw/main/Images/Predit-all.png" alt="Predict All Users Page" />
    </td>
    <td>
      <img src="https://github.com/jejo205713/neural-credit-scoring/raw/main/Images/Add-new-user.png" alt="Add New User Page" />
    </td>
  </tr>
</table>

---

## ✨ Features

-   **🏠 Professional Dashboard:** A clean, minimal, and responsive user interface built with Flask, featuring a white, red, and maroon theme.
-   **🔧 Interactive Model Training:** Train a `RandomForestClassifier` on your dataset with a single click. After training, the dashboard automatically displays:
    -   **Feature Importance Chart:** Understand which data points most influence the model's decisions.
    -   **Confusion Matrix:** Visually assess the model's accuracy, precision, and recall.
-   **👤 Single User Prediction & Analysis:**
    -   Select any user from the production dataset.
    -   Receive an instant risk classification ("Low Risk" or "High Risk").
    -   View the model's prediction confidence score.
    -   Analyze an interactive bar chart comparing the user's features against the average profiles of "Low Risk" and "High Risk" cohorts.
-   **👥 Batch Prediction:** Run the model on the entire production dataset and view all predictions in a clean, formatted table with color-coded risk tags.
-   **➕ Data Management:** Easily add new user data to the production dataset through a user-friendly form with sensible defaults.

---

## 🛠️ Tech Stack

-   **Backend:** Python, Flask
-   **Frontend:** HTML, CSS, JavaScript
-   **Machine Learning:** Scikit-learn
-   **Data Manipulation:** Pandas, NumPy
-   **Data Visualization:** Plotly.js
-   **Explainability (Future):** SHAP, NetworkX, Node2Vec

---

## 📁 Project Structure

The repository is organized to separate concerns, making it scalable and easy to maintain.

```
/credit-prototype/
|
├── app4.py                  # Main Flask application logic
├── requirements.txt         # Project dependencies
|
├── data/                    # Datasets
│   ├── production/
│   │   └── neural_credit_data.csv
│   └── training/
│       └── credit_dataset.csv
|
├── models/                  # Saved machine learning models
│   └── credit_model.pkl
|
├── preprocessing/           # Data preprocessing scripts
│   └── data_preprocessor.py
|
├── static/                  # CSS stylesheets
│   └── css/
│       └── style.css
|
└── templates/               # HTML files for the web interface
    ├── layout.html          # Base template with sidebar
    ├── index.html           # Overview page
    ├── train.html           # Model training page
    ├── predict_user.html    # Single user prediction page
    ├── predict_all.html     # Batch prediction page
    └── add_user.html        # New user form
```

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

-   Git
-   Python 3.9 or higher

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone <your-repository-url>
    cd credit-prototype
    ```

2.  **Create and activate a virtual environment:**
    -   **Linux/macOS:**
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
    -   **Windows:**
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Prepare your data:**
    -   Ensure your training data is present at `data/training/credit_dataset.csv`.
    -   Ensure your production data file exists at `data/production/neural_credit_data.csv`. If the file is empty, it must at least contain the headers.

### Running the Application

1.  **Start the Flask server:**
    ```sh
    python app4.py
    ```

2.  **Open the dashboard:**
    Navigate to `http://127.0.0.1:5000` in your web browser.

---

## 🕹️ How to Use the Dashboard

1.  **Train the Model:** The first time you run the application, the model will not be trained. Navigate to the **Train Model** page and click the "Start Training" button. This will train the model and save it to the `models/` directory.
2.  **Predict a User:** Go to the **Predict User** page, select a username from the dropdown, and click "Predict" to see the detailed analysis.
3.  **Explore Other Features:** Use the sidebar to navigate between batch predictions and adding new users.

---

## 📈 Future Work

This platform is designed to be extensible. Potential future enhancements include:
-   **Integrate SHAP:** Activate the `shap_explainer.py` to provide detailed, instance-level explanations for each prediction.
-   **Activate LLM Explanations:** Implement the `mini_llm_explainer.py` to generate natural language summaries of prediction results.
-   **User Authentication:** Add a login system to secure the dashboard.
-   **Database Integration:** Replace the CSV files with a robust database (like PostgreSQL or SQLite) for better data management.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
