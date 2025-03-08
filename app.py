import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

# 🔹 Load trained models
kmeans = joblib.load("kmeans_tsne.pkl")
tsne_model = joblib.load("tsne_transform.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# 🔹 Streamlit UI
st.title("🐾 Animal Disease Prediction System")
st.write("🚀 Enter details to predict the most likely disease cluster.")

# 🔹 Step 1: Animal Type Selection
animal_type = st.selectbox("Select Animal Type", ["Cow", "Dog", "Cat", "Horse"])

# 🔹 Step 2: Breed Selection
breed_options = {
    "Dog": ["Labrador", "Beagle", "German Shepherd", "Poodle"],
    "Cat": ["Persian", "Siamese", "Maine Coon", "Bengal"],
    "Horse": ["Thoroughbred", "Arabian", "Mustang", "Clydesdale"],
    "Cow": ["Holstein", "Jersey", "Angus", "Hereford"]
}
breed = st.selectbox("Select Breed", breed_options[animal_type])

# 🔹 Step 3: Collect Numeric Inputs
age = st.slider("Age of the Animal", 0, 30, 5)
weight = st.number_input("Weight (kg)", min_value=1, max_value=500, value=50)
temperature = st.number_input("Body Temperature (°C)", min_value=30, max_value=45, value=38)

# 🔹 Step 4: Symptoms Input
symptoms = st.multiselect(
    "Select Symptoms",
    ["Vomiting", "Diarrhea", "Coughing", "Lameness", "Skin_Lesions", "Nasal Discharge", "Eye Discharge"]
)

# 🔥 Encode categorical features
try:
    animal_type_encoded = label_encoders["Animal_Type"].transform([animal_type])[0]
    breed_encoded = label_encoders["Breed"].transform([breed])[0]
except KeyError as e:
    st.error(f"❌ Encoding Error: {e}")
    st.stop()

# 🔹 Convert Symptoms to Binary Format
symptom_features = ["Vomiting", "Diarrhea", "Coughing", "Lameness", 
                    "Skin_Lesions", "Nasal_Discharge", "Eye_Discharge"]
symptom_values = [1 if symptom in symptoms else 0 for symptom in symptom_features]

# 🔹 Create input DataFrame
input_data = pd.DataFrame([[animal_type_encoded, breed_encoded, age, weight, temperature] + symptom_values],
                          columns=["Animal_Type", "Breed", "Age", "Weight", "Body_Temperature"] + symptom_features)

# 🔹 Ensure input_data has all features expected by the scaler
expected_columns = scaler.feature_names_in_

# 🔥 Add missing columns (Clusters) with default values
missing_columns = set(expected_columns) - set(input_data.columns)
for col in missing_columns:
    input_data[col] = 0  # Default value for missing cluster features

# 🔹 Reorder columns to match the scaler’s order
input_data = input_data[expected_columns]

# 🔹 Convert input to float before scaling
input_data = input_data.astype(np.float64)

# 🔹 Apply scaling safely
try:
    input_scaled = scaler.transform(input_data)
except ValueError as e:
    st.error(f"❌ Data Scaling Failed: {e}")
    st.stop()

# 🔹 Apply t-SNE transformation
if input_scaled.shape[0] < 2:
    st.warning("⚠️ Skipping t-SNE: Not enough samples. Using a default placeholder.")
    input_tsne = np.zeros((1, 2))  # Fallback to 2D zero array
else:
    perplexity_value = min(5, input_scaled.shape[0] - 1)
    input_tsne = tsne_model.fit_transform(input_scaled)

# 🔹 Ensure input_tsne is properly reshaped for prediction
if input_tsne.ndim == 1:
    input_tsne = input_tsne.reshape(1, -1)

# 🔹 Predict Cluster
cluster = kmeans.predict(input_tsne)[0]

# 🔹 Display Prediction
st.success(f"✅ Based on the given information, the predicted disease falls under **Cluster {cluster}**.")
