
       import gradio as gr
import joblib
import numpy as np

model = joblib.load("heart_disease_model.pkl")

def predict(age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal):

    data = np.array([[age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        return "❤️ Heart Disease Detected"
    else:
        return "✅ No Heart Disease Detected"

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Sex (0=Female, 1=Male)"),
        gr.Number(label="Chest Pain Type"),
        gr.Number(label="Resting Blood Pressure"),
        gr.Number(label="Cholesterol"),
        gr.Number(label="Fasting Blood Sugar"),
        gr.Number(label="Rest ECG"),
        gr.Number(label="Max Heart Rate"),
        gr.Number(label="Exercise Angina"),
        gr.Number(label="Oldpeak"),
        gr.Number(label="Slope"),
        gr.Number(label="CA"),
        gr.Number(label="Thal")
    ],
    outputs="text",
    title="Heart Disease Prediction System"
)

interface.launch()
