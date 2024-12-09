import streamlit as st
import google.generativeai as genai
import os
import torch
from dotenv import load_dotenv
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from io import BytesIO
from predictor import *
from image_processor import *
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import google.generativeai as genai
import gdown
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

gen_model = genai.GenerativeModel('gemini-1.5-flash-latest')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@st.cache_resource
def download_model():
    file_id = "1Ovlm72q3sa6BxobWb-6QKTAF-uv753kZ"
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    gdown.download(url, 'model.pth', quiet=False)
    model = models.vgg16(weights = 'VGG16_Weights.DEFAULT')
    model.classifier[6] = nn.Linear(4096,4)
    return model

model = download_model()

def createForm(prediction):
    st.markdown("<h2 style='text-align: center;'>To get the report fill the details</h2>", unsafe_allow_html=True)
    with st.form(key='user_info_form'):
        patient_name = st.text_input("Patient Name")
        patient_age = st.number_input("Age", min_value=0, max_value=120)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        patient_symptoms = st.text_area("Other Symptoms", placeholder="Describe the symptoms...")
        submit_button = st.form_submit_button("Generate Report")
    if submit_button:
        if not patient_name:
            st.error('Patient name is required!!')
        elif not patient_age:
            st.error('Patient age is required!!')
        if patient_symptoms == "":
            patient_symptoms = 'NIL'
        user_data = {
            "Patient Name": patient_name,
            "Age": patient_age,
            "Gender": patient_gender,
            "Symptoms": patient_symptoms
        }
        st.markdown(f"### Report for {patient_name}")
        st.write("(The report generation will be solely based on the symptoms provided and prediction)")
        generate_and_display_report(prediction, user_data)
        
def generate_report(prediction,patient_details):
    prompt = f"""You have to generate a medical report based on the predicted brain tumor by MRI: {prediction} and the patient details provided below.
                ### Patient details:
                1. **Patient Name: {patient_details["Patient Name"]}**
                2. **Patient Age: {patient_details["Age"]}**
                3. **Patient Gender: {patient_details["Gender"]}**
                4. **Patient Symptoms: {patient_details["Symptoms"]}**

                ### Report Instructions:
                1. **Include the patient details in the header without patient symptoms** as listed above. Each piece of information must be on a separate line (e.g., "Patient Name" on its own line, followed by "Patient Age" on its own line).
                2. **Do not place any of the details on the same line.** Each detail must appear separately as shown in the list.
                3. After the patient details, generate a medical report in subsections with each section containing a maximum of 5 lines:
                    - Diagnosis
                    - Possible Cause of the condition based on patient details
                    - Treatment options and recommendations
                    - Prognosis
                4. If Patient Symptoms is not empty then analyse those symptoms in diagnosis.
                5. **Strictly follow these formatting rules**: 
                    - No bullet points or extra punctuation other than what's necessary for a medical report.
    """

    response = gen_model.generate_content(
        contents=prompt
    )
    
    if response._done and response._result and 'candidates' in response._result:
        report_content = response.text
        return report_content
    else:
        return "Error: Report generation failed."

def generate_and_display_report(prediction,patient_details):
    report = generate_report(prediction, patient_details)
    st.markdown("<h2 style='text-align: center; background-color: #17253b'>Generated Report</h2>", unsafe_allow_html=True)
    st.write(report)
    st.download_button(
        label="Download Text Report",
        data=report,
        file_name="generated_report.txt",
        mime="text/plain"
    )

def stats(logits):
    
    probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()
    if probabilities.ndim > 1:
        probabilities = probabilities[0]
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    probabilities_normalized = probabilities / np.sum(probabilities)
    percentages = np.round(probabilities_normalized * 100, 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = plt.Normalize(vmin=np.min(probabilities_normalized), vmax=np.max(probabilities_normalized))
    colors = plt.cm.coolwarm(norm(probabilities_normalized))
    sns.barplot(x=probabilities_normalized, y=class_labels, palette=colors, orient='h', width=0.2, ax=ax)
    plt.title("Probabilities Window")
    plt.xlabel("Probability")
    plt.ylabel("Predictions")
    for i, percentage in enumerate(percentages):
        plt.text(probabilities_normalized[i] + 0.02, i, f'{percentage}%', va='center', fontsize=12)
    st.pyplot(fig)
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;  /* Set text color to white for visibility */
    }
    .stApp {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
        .stForm {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .stDownloadButton button{
            color: #eb4634;
            border: 1px solid #eb4634
            border-radius: 8px;      
        }
        .stFormSubmitButton button{
            color: #eb4634;
            border: 1px solid #eb4634
            border-radius: 8px;    
        }
    </style>
""", unsafe_allow_html=True)

st.title("Brain Tumor Classification")
uploaded_file = st.file_uploader("Upload an MRI Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    prediction, outputs = predict(uploaded_file, model, device)
    image = Image.open(uploaded_file).convert('RGB')
    saliency_image, blue_image, grad_image = processor(model, uploaded_file, device)
    # st.image(image, caption="Uploaded Image", width=300)
    fig, axes = plt.subplots(1,2, figsize=(15,5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image" , color="white")
    axes[0].axis('off')
    axes[1].imshow(blue_image)
    axes[1].set_title("BW transformation", color="white")
    axes[1].axis('off')
    fig.patch.set_facecolor("black")
    st.pyplot(fig)
    fig1, axes1 = plt.subplots(1,2, figsize=(15,5))

    axes1[0].imshow(grad_image)
    axes1[0].set_title("GRAD transformation", color="white")
    axes1[0].axis('off')
    axes1[1].imshow(saliency_image)
    axes1[1].set_title("SALIENT", color="white")
    axes1[1].axis('off')
    fig1.patch.set_facecolor("black")
    st.pyplot(fig1)
    st.markdown(
    f"""
    <div style='
        margin-top: 20px; 
        padding: 15px; 
        background-color: #28292b; 
        color: white; 
        border-radius: 5px; 
        text-align: left; 
        display: inline-block;
        width: 100%;'>
        <h2 style='margin: 0; padding: 0;'>Prediction: <b>{prediction}</b></h2>
    </div>
    """,
    unsafe_allow_html=True
)
    # logits = torch.randn(1, 4)  
    st.markdown("<h1 style='text-align: center;'>Analysis</h1>", unsafe_allow_html=True)
    stats(outputs)
    # Report Section
    createForm(prediction)

