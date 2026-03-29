"""
train_model.py
--------------
Trains and saves the ML + Neural Network models for the AI Skill Suggester.
Run this ONCE before launching the Streamlit app:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Try to import Keras (TensorFlow)
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    USE_NEURAL_NET = True
except ImportError:
    print("TensorFlow not found. Only the ML model will be trained.")
    USE_NEURAL_NET = False

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("skills_dataset.csv")
print(f"   Columns found: {list(df.columns)}")

# 2. Encode difficulty
difficulty_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
df["difficulty_encoded"] = df["difficulty"].map(difficulty_map)

# 3. Encode cost
cost_map = {"Free": 0, "Paid": 1}
df["cost_encoded"] = df["cost_level"].map(cost_map)

# 4. Encode category
le_category = LabelEncoder()
df["category_encoded"] = le_category.fit_transform(df["category"])

# 5. Encode skill names (target label)
le_skill = LabelEncoder()
df["skill_encoded"] = le_skill.fit_transform(df["skill_name"])

# 6. Encode interest tags
print("Engineering features...")

# Clean the interest_tags column first
df["interest_tags"] = df["interest_tags"].fillna("").astype(str)

# Split and clean each row's tags
def clean_tags(tag_string):
    return [t.strip() for t in tag_string.split("/") if t.strip()]

tag_lists = df["interest_tags"].apply(clean_tags)

mlb = MultiLabelBinarizer()
tags_matrix = mlb.fit_transform(tag_lists)
print(f"   Tag classes found: {list(mlb.classes_)}")

# Build tags dataframe with clean column names
tag_col_names = [f"tag_{t}" for t in mlb.classes_]
tags_df = pd.DataFrame(tags_matrix, columns=tag_col_names)

# Reset index before concat to avoid alignment issues
df = df.reset_index(drop=True)
tags_df = tags_df.reset_index(drop=True)

# Add tag columns to main dataframe one by one
for col in tag_col_names:
    df[col] = tags_df[col]

# 7. Build feature matrix using ONLY the columns we want
feature_cols = ["difficulty_encoded", "cost_encoded", "weekly_hours_needed"] + tag_col_names

print(f"   Feature columns: {feature_cols}")

# Select only those columns and convert to float
X_df = df[feature_cols].copy()
print(f"   Data types:\n{X_df.dtypes}")

X = X_df.values.astype(float)
y = df["skill_encoded"].values

print(f"   Features: {X.shape[1]}  |  Samples: {X.shape[0]}")

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Train Random Forest
print("\nTraining Random Forest classifier...")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print(f"   Random Forest Accuracy: {(rf_preds == y_test).mean():.2%}")

# 10. Train Neural Network
nn_model = None
if USE_NEURAL_NET:
    print("\nTraining Neural Network...")
    num_classes = len(le_skill.classes_)

    nn_model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="SkillSuggesterNN")

    nn_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = nn_model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=0
    )

    val_acc = max(history.history["val_accuracy"])
    print(f"   Neural Network Best Val Accuracy: {val_acc:.2%}")
    nn_model.save("nn_model.keras")
    print("   Neural network saved -> nn_model.keras")

# 11. Save everything
print("\nSaving models and encoders...")
artifacts = {
    "rf_model": rf_model,
    "le_skill": le_skill,
    "le_category": le_category,
    "mlb": mlb,
    "feature_cols": feature_cols,
    "difficulty_map": difficulty_map,
    "cost_map": cost_map,
    "use_neural_net": USE_NEURAL_NET,
}
with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("   Encoders + RF model saved -> model_artifacts.pkl")
print("\nTraining complete! You can now run:  streamlit run app.py")
