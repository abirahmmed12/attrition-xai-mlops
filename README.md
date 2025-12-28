# 📉 Employee Attrition Prediction with MLOps & Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![MLOps](https://img.shields.io/badge/Tools-MLflow%20%7C%20DVC-orange)
![Model](https://img.shields.io/badge/Model-TabPFN-red)

## 🚀 Project Overview
This project is an end-to-end **Machine Learning system** designed to predict employee attrition (whether an employee will leave or stay). It goes beyond simple prediction by using **Explainable AI (XAI)** logic to provide actionable insights into *why* an employee might leave.

The system is built using a **Modular MLOps Architecture**, ensuring scalability, reproducibility, and easy deployment. It uses **TabPFN**, a state-of-the-art transformer-based model optimized for tabular data.

## ✨ Key Features
* **🎯 High Accuracy:** Powered by **TabPFN** (Transformer-based) model, ideal for small to medium datasets.
* **🧠 Explainable AI (XAI):** Provides real-time reasons for attrition (e.g., "Low Salary", "Lack of Promotion").
* **⚡ Real-time Inference:** Optimized pipeline for sub-second prediction latency.
* **🛠 MLOps Integration:**
    * **MLflow:** For experiment tracking and model versioning.
    * **DVC:** For data version control.
    * **Modular Code:** Clean, production-ready code structure.
* ** Interactive UI:** User-friendly web interface built with **Flask**, HTML, and CSS.

## 🛠️ Tech Stack
* **Language:** Python
* **Web Framework:** Flask
* **Machine Learning:** TabPFN, Scikit-learn, Pandas, NumPy
* **MLOps Tools:** MLflow, DVC
* **Frontend:** HTML5, CSS3, Bootstrap
* **Version Control:** Git & GitHub

## 📂 Project Structure
```bash
├── artifacts/          # Stores trained models & preprocessors
├── data-source/        # Raw and processed datasets
├── logs/               # System logs for debugging
├── mlruns/             # MLflow experiment tracking logs
├── src/                # Source code for pipelines
│   ├── components/     # Data ingestion, transformation, training
│   ├── pipeline/       # Training and Prediction pipelines
│   ├── utils.py        # Utility functions
├── templates/          # HTML files for the web UI
├── app.py              # Main Flask application entry point
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation



⚙️ Installation & Setup

1. Clone the Repository
git clone [https://github.com/abirahmmed12/attrition-xai-mlops.git](https://github.com/abirahmmed12/attrition-xai-mlops.git)
cd attrition-xai-mlops

2. Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application
python main.py


How It Works
Data Input: User enters employee details (Age, Salary, Job Role, etc.) via the Web UI.

Preprocessing: The system processes data using a saved preprocessor pipeline.

Prediction: The pre-loaded TabPFN model predicts the probability of attrition.

XAI Analysis: A custom logic engine analyzes feature contributions to generate "Risk Factors" or "Retention Drivers".

Output: The UI displays the result (Leave/Stay) along with specific reasons and confidence scores.

<img width="733" height="910" alt="image" src="https://github.com/user-attachments/assets/9aa90586-17e1-4616-983e-04b8f817e85b" />

🤝 Contribution
Contributions are welcome! Feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License.

Developed by Abir Ahmmed
