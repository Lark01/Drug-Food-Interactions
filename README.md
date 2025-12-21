# 💊 Drug–Food Interaction Detector

A Machine Learning application designed to flag potential adverse interactions between prescribed drugs and meals. This project combines computer vision for food recognition with a structured machine learning pipeline to analyze nutritional content and predict interaction risks.

> **Note:** The tabular dataset used for training the interaction model was synthetically generated/sourced via ChatGPT due to resource limitations.

## 🚀 Features

- **Food Recognition**: Upload a photo of your meal, and the app identifies the food item using a custom Keras model trained on the Food-101 dataset.
- **Drug Selection**: Choose from a list of common prescribed medications (e.g., Warfarin, Metformin, etc.).
- **Interaction Analysis**: The system estimates the nutritional profile of the detected food and runs it through an SVM-based interaction model to predict potential risks.
- **User-Friendly Interface**: Built with Streamlit for an easy-to-use web experience.

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Web Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**:
  - **Scikit-learn**: For the interaction prediction pipeline (SVM).
  - **TensorFlow/Keras**: For the food image classification model.
- **Data Processing**: Pandas, NumPy
- **Deployment**: Docker

## 📂 Project Structure

```
Drug-Food-Interactions/
├── DrugInteraction/           # Data Analysis & Interaction Model Training
│   ├── DrugInteractionNotebook.ipynb  # EDA, Feature Engineering, SVM Training
│   └── Drugs_dataset.csv              # Tabular dataset for drug-food interactions
│
├── FoodDetection/             # Food Image Classification
│   ├── food101-training.ipynb         # Training notebook for Food-101 model
│   ├── food101-mobilenetv3.ipynb      # Training notebook for Food-101 using MobileNetV3
│   └── food101_custom_FINAL_51acc.keras # Trained Keras model (Source)
│
├── ML_Deployment/             # Deployment Artifacts
│   ├── deploy.py                      # Main Streamlit application
│   ├── Dockerfile                     # Docker configuration
│   ├── requirements.txt               # Python dependencies
│   ├── svm_full_pipeline.joblib       # Serialized Interaction Model Pipeline
│   └── food101_custom_FINAL_51acc.keras # Copy of the image model for deployment
│
└── README.md                  # Project Documentation
```

## 💻 Quick Start (Local)

### Prerequisites
- Python 3.9 or higher installed.

### Installation

1.  **Clone the repository** (if applicable) or navigate to the project folder.
2.  **Navigate to the deployment directory**:
    ```bash
    cd ML_Deployment
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Verify Model Files**:
    Ensure the following files are present in the `ML_Deployment` folder:
    - `svm_full_pipeline.joblib`
    - `food101_custom_FINAL_51acc.keras`

### Running the App

Run the Streamlit application:
```bash
streamlit run deploy.py
```
The app will typically be accessible at `http://localhost:8501` or `http://localhost:7860`.

## 🐳 Docker Deployment

You can containerize and run the application using Docker.

1.  **Build the image**:
    ```bash
    docker build -t drug-food-app ML_Deployment
    ```

2.  **Run the container**:
    ```bash
    docker run -p 7860:7860 drug-food-app
    ```
    Access the app at `http://localhost:7860`.

## 🧠 Model Details

### 1. Food Classification Model
- **Architecture**: Custom CNN / Transfer Learning (based on Food-101 training).
- **File**: `food101_custom_FINAL_51acc.keras`
- **Function**: Takes a user-uploaded image and predicts the food class (e.g., "Pizza", "Salad").

### 2. Interaction Prediction Model
- **Architecture**: Support Vector Machine (SVM) within a Scikit-learn Pipeline.
- **File**: `svm_full_pipeline.joblib`
- **Pipeline Steps**:
  - **Preprocessing**: Log transformation ($\log(1+x)$) for numeric columns to reduce skew.
  - **Encoding**: Target encoding for Drug names.
  - **Feature Engineering**: Creation of interaction terms between drug properties and food nutrients.
  - **Prediction**: Ternary classification of interaction risk:
  0: None
  1: Minor
  2: Major

## 📝 Future Improvements
- **Data Validation**: Add tests for preprocessing transformers to guard against schema drift.
- **Nutrient API**: Replace the static nutrition lookup with a real-time API (e.g., USDA FoodData Central).
- **Model Robustness**: Retrain the interaction model on real-world clinical data if available.

