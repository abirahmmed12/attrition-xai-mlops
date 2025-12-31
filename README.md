<div align="center">

# 📉 Employee Attrition Prediction with MLOps & XAI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-Flask-green?style=for-the-badge&logo=flask&logoColor=white)
![MLOps](https://img.shields.io/badge/Tools-MLflow%20%7C%20DVC-orange?style=for-the-badge&logo=mlflow&logoColor=white)
![Model](https://img.shields.io/badge/Model-TabPFN-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<p align="center">
  <b>An end-to-end Machine Learning system designed to predict employee attrition with explainable insights.</b>
</p>

</div>

---

## 📋 Table of Contents
- [🚀 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#-tech-stack)
- [⚙️ System Workflow](#-system-workflow)
- [📂 Project Structure](#-project-structure)
- [📸 UI Preview](#-ui-preview)
- [💻 Installation & Setup](#-installation--setup)
- [🤝 Contribution](#-contribution)
- [📜 License](#-license)

---

## 🚀 Project Overview

This project is a production-ready **Machine Learning system** designed to predict whether an employee is likely to leave the organization. Unlike traditional black-box models, this system integrates **Explainable AI (XAI)** to provide actionable insights into *why* an attrition risk exists (e.g., "Low Salary", "Lack of Promotion", "Overtime Stress").

Built with a **Modular MLOps Architecture**, it ensures scalability, reproducibility, and seamless deployment. The core predictive engine is powered by **TabPFN**, a state-of-the-art transformer-based model optimized for tabular data, making it highly effective even for small to medium-sized datasets typical in SMEs.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **🎯 High Accuracy** | Powered by the **TabPFN** (Transformer-based) model, outperforming traditional algorithms on small datasets. |
| **🧠 Explainable AI** | Provides real-time reasons for attrition, categorizing factors into "Risk Drivers" and "Retention Drivers". |
| **⚡ Real-time Inference** | Optimized prediction pipeline ensuring sub-second latency for instant feedback. |
| **🛠 MLOps Ready** | Integrated with **MLflow** for experiment tracking and **DVC** for data version control. |
| **💻 Interactive UI** | User-friendly web interface built with **Flask**, HTML5, and Bootstrap 5. |
| **🐳 Containerized** | (Optional) Ready for Docker deployment for consistent environments. |

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Programming Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) |
| **Web Framework** | ![Flask](https://img.shields.io/badge/-Flask-000000?logo=flask&logoColor=white) |
| **Machine Learning** | TabPFN, Scikit-learn, Pandas, NumPy |
| **MLOps Tools** | ![MLflow](https://img.shields.io/badge/-MLflow-0194E2?logo=mlflow&logoColor=white) ![DVC](https://img.shields.io/badge/-DVC-945DD6?logo=dvc&logoColor=white) |
| **Frontend** | HTML5, CSS3, Bootstrap 5 |
| **Version Control** | ![Git](https://img.shields.io/badge/-Git-F05032?logo=git&logoColor=white) |

---

## ⚙️ System Workflow

The application follows a linear inference pipeline:

```mermaid
graph LR
    A[User Input via UI] --> B[Data Preprocessing]
    B --> C[TabPFN Model Inference]
    C --> D{Prediction?}
    D -->|Probability Score| E[XAI Logic Engine]
    E --> F[Generate Risk Factors]
    F --> G[Final Output on Dashboard]

    📂 Project Structure

    attrition-xai-mlops/
├── artifacts/          # 📦 Stores trained models (model.pkl) & preprocessors
├── data-source/        # 📊 Raw and processed datasets (CSV)
├── logs/               # 📝 System logs for debugging and monitoring
├── mlruns/             # 📉 MLflow experiment tracking logs
├── src/                # 🧠 Source code for the ML pipeline
│   ├── components/     # Modules for Ingestion, Transformation, & Training
│   ├── pipeline/       # Orchestration for Training and Prediction
│   ├── utils.py        # Helper functions (save/load objects)
│   └── logger.py       # Custom logging configuration
├── templates/          # 🎨 HTML files for the Web UI
├── static/             # 🖼️ CSS, Images, and JavaScript assets
├── main.py             # 🚀 Main Flask application entry point
├── requirements.txt    # 📋 List of project dependencies
└── README.md           # 📖 Project documentation


📸 UI Preview
Here is a glimpse of the prediction dashboard:

<div align="center"> <img src="https://github.com/user-attachments/assets/9aa90586-17e1-4616-983e-04b8f817e85b" alt="App Screenshot" width="700"> </div>

💻 Installation & Setup
Follow these steps to run the project locally.

1. Clone the Repository
git clone [https://github.com/abirahmmed12/attrition-xai-mlops.git](https://github.com/abirahmmed12/attrition-xai-mlops.git)
cd attrition-xai-mlops

2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.
# For Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# For Windows
python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application
Start the Flask server:
python main.py
Open your browser and navigate to: http://127.0.0.1:5000

🤝 Contribution
Contributions are welcome! If you have suggestions for improvements or bug fixes:

Fork the repository.

Create a new branch (git checkout -b feature-branch).

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature-branch).

Open a Pull Request.

📜 License
This project is licensed under the MIT License.

<div align="center"> <b>Developed by <a href="https://www.google.com/search?q=https://github.com/abirahmmed12">Abir Ahmmed</a></b> </div>