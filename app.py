import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 🔹 Load trained models
kmeans = joblib.load("kmeans_tsne.pkl")
tsne_model = joblib.load("tsne_transform.pkl")
scaler = joblib.load("scaler.pkl")

# 🔹 Streamlit UI
st.title("🐾 Animal Disease Prediction System")
st.write("🚀 Enter details to predict the most likely disease.")

# 🔹 Step 1: Animal Type Selection
animal_type = st.selectbox("Select Animal Type", ["Cow", "Dog", "Cat", "Horse"])

# 🔹 Step 2: Breed Selection
if animal_type == "Dog":
    breed = st.selectbox("Select Breed", ["Labrador", "Beagle", "German Shepherd", "Poodle"])
elif animal_type == "Cat":
    breed = st.selectbox("Select Breed", ["Persian", "Siamese", "Maine Coon", "Bengal"])
elif animal_type == "Horse":
    breed = st.selectbox("Select Breed", ["Thoroughbred", "Arabian", "Mustang", "Clydesdale"])
else:
    breed = st.selectbox("Select Breed", ["Holstein", "Jersey", "Angus", "Hereford"])

# 🔹 Step 3: Collect Demographic Info
age = st.slider("Age of the Animal", 0, 30, 5)
weight = st.number_input("Weight (kg)", min_value=1, max_value=500, value=50)
temperature = st.number_input("Body Temperature (°C)", min_value=30, max_value=45, value=38)

# 🔹 Step 4: Symptoms Input
symptoms = st.multiselect(
    "Select Symptoms",
    ["Vomiting", "Diarrhea", "Coughing", "Lameness", "Skin Lesions", "Nasal Discharge", "Eye Discharge"]
)

# 🔹 Only process when the button is clicked
if st.button("🔍 Predict Disease"):
    try:
        # Convert Input to DataFrame
        input_data = pd.DataFrame([[animal_type, breed, age, weight, temperature] + symptoms])

        # 🔹 Standardize Input
        input_scaled = scaler.transform(input_data)

        # 🔹 Apply t-SNE transformation
        if input_scaled.shape[0] == 1:
            input_tsne = tsne_model.fit_transform(np.vstack([input_scaled, input_scaled]))[:1]
        else:
            input_tsne = tsne_model.fit_transform(input_scaled)

        # 🔹 Predict Cluster
        cluster = kmeans.predict(input_tsne)[0]

        # 🔹 Debugging: Print cluster value
        st.write(f"🔍 Debug: Predicted Cluster: {cluster}")

        # 🔹 Define Disease Mapping for Clusters
        disease_mapping = {
            0: "Canine Distemper",
            1: "Feline Upper Respiratory Infection",
            2: "Equine Influenza",
            3: "Bovine Respiratory Disease",
            4: "Canine Parvovirus",
            5: "Lyme Disease",
            6: "Salmonella Infection"
        }

        # 🔹 Get the Predicted Disease Name
        predicted_disease = disease_mapping.get(cluster, "Unknown Disease")

        # 🔹 Display Prediction
        st.success(f"✅ Based on the given information, the predicted disease is **{predicted_disease}**.")

    except ValueError as e:
        st.error(f"❌ Error in Prediction: {e}")
