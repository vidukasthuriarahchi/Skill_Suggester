# What Can I Learn Today? - AI Skill Suggester
Final Project | AI Essentials: From Data to Agents
By Thenuji Vidunima Kasthuriarachchi

---

## How to Run (Step by Step)

### Step 1 - Install Python
Make sure you have Python 3.9+ installed.
Download from: https://www.python.org/downloads/

### Step 2 - Install dependencies
Open your terminal/command prompt, navigate to this folder, and run:

    pip install -r requirements.txt

### Step 3 - Train the model (run ONCE)

    python train_model.py

This creates two files:
- model_artifacts.pkl  (encoders + Random Forest model)
- nn_model.keras       (trained Neural Network)

### Step 4 - Launch the app

    streamlit run app.py

Your browser will open automatically at http://localhost:8501

---

## Project Structure

    skill_suggester/
    |-- app.py               Main Streamlit web application
    |-- train_model.py       Model training script (run once)
    |-- skills_dataset.csv   40-skill dataset
    |-- requirements.txt     Python dependencies
    |-- README.md            This file

---

## How the AI Works

1. Data          - 40 skills across 10 categories with features
2. Preprocessing - Label encoding for difficulty/cost, tag binarization
3. ML Model      - Random Forest (200 trees) for classification
4. Neural Net    - 4-layer MLP built with Keras/TensorFlow
5. Dashboard     - Streamlit app with recommendations and charts
