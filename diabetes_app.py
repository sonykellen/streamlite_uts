import streamlit as st # type: ignore
import pandas as pd # type: ignore
from sklearn.tree import DecisionTreeClassifier


# Memuat dataset dan melatih model
df = pd.read_csv('diabetes.csv')  # Ganti dengan path dataset kamu
X = df.drop('Outcome', axis=1)
y = df['Outcome']
model = DecisionTreeClassifier()
model.fit(X, y)

# Fungsi untuk memprediksi diabetes
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    prediction = model.predict(input_data)
    return "Positif" if prediction[0] == 1 else "Negatif"

# Membuat UI dengan Streamlit
st.title('Deteksi Penyakit Diabetes dengan Decision Tree')
st.write("Masukkan data berikut untuk mendeteksi apakah pasien berisiko terkena diabetes:")

# Membagi kolom
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)

with col1:
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)

with col1:
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=80)

with col1:
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)


with col2:
    insulin = st.number_input('Insulin', min_value=0, max_value=900, value=30)

with col2:
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)

with col2:    
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)

with col2:
    age = st.number_input('Age', min_value=0, max_value=120, value=30)

if st.button('Prediksi Diabetes'):
    result = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    st.write(f"Hasil Prediksi: {result}")
