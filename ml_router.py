import json
import os
import pdb
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import PIL
import requests
from diskcache import Cache
from openai import OpenAI
from PIL import Image
from termcolor import colored
import numpy as np

import joblib
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

import requests



def mlClassifier(doc):
    
    # api_key = "......................."
    # os.environ['OPENAI_API_KEY'] = api_key

    llm = ChatOpenAI(model_name='gpt-4', temperature=0)

    template = """
        As a classifier, your task is to analyze text inputs and determine their dominant intent based on whether 
        they refer to a request or information related to ML model or LLM Model. When inputs contain multiple 
        elements, focus on the final action or the most dominant element of the request. Assign a numerical 
        code based on these categories:

        - 0: Non ML (For health insurance general questions or creating an image of something or any other general question)
        - 1: ML (for disease prediction for health insurance plan recommendation)
        

        Given the input text below, classify it according to the dominant intent or final action requested:

        Examples:

        "input" : "I want recommendation for a heath insurance plan", "output": "1",
        "input" : "Suggest me health insurance plans", "output": "1",
        "input" : "Recommend insurance plan to me", "output": "1",
        "input" : "Can you help me understand the health insurance plans based on conditions", "output": "1",
        "input" : "I am experiencing health related issues which health insurance plan would be good for me", "output": "1",
        "input" : "What health insurance plan would cover treatments, can you suggent me that plan?", "output": "1",
        "input" : "Can you suggest a health insurance option?", "output": "1",
        "input" : "recommend me a plan for general health insurance", "output": "1",
        "input" : "Can you recommend a health insurance plan that covers treatments for persistent cough and shortness of breath?", "output": "1",
        "input" : "Which health insurance plan would be suitable for someone experiencing fever, body aches, and fatigue?", "output": "1",
        "input" : "Can you recommend a insurance plan that includes coverage for sudden weight loss and loss of appetite?", "output": "1",
        "input" : "What health insurance plan would cover treatments for persistent diarrhea with blood and mucus?", "output": "1",
        "input" : "Can you recommend a health insurance plan that includes coverage for frequent fevers with chills and sweating?", "output": "1",
        "input" : "Can you recommend a health insurance plan that includes coverage for frequent urination, thirst, and fatigue?", "output": "1",
        "input" : "What insurance plan would cover treatments for difficulty breathing, wheezing, and chest tightness?", "output": "1",
        
        "input" : "create an image of a deer", "output": "0", 
        "input" : "I am not well", "output": "0"
        "input" : "I am not feeling that well what to do", "output": "0"
        "input" : "symptoms", "output": "0",
        "input" : "medical conditions", "output": "0",
        "input" : "Can you clarify the concept of co-payments and coinsurance in relation to health insurance coverage?", "output": "0",
        "input" : "specific symptoms", "output": "0",
        "input" : "disease diagnosis", "output": "0",
        "input" : "disease prediction", "output": "0",
        "input" : "What could be causing my persistent cough and shortness of breath?", "output": "0",
        "input" : "My throat is sore, and I have difficulty swallowing, what might be the issue?", "output": "0",
        "input" : "Which plan would cover treatments for skin rashes, itching, and swelling?", "output": "0",
        "input" : "What are the key factors to consider when selecting a health insurance plan?", "output": "0",
        "input" : "What options are available for adding dependents to my health insurance policy, and how does it impact my premiums?", "output": "0",
        "input" : "What are the coverage options someone has under Medicare Part B to cover healthcare and drug costs not covered by Medicare itself?", "output": "0",
        "input" : "What state agency can someone contact if they need help understanding their health insurance options after losing Medicaid coverage?", "output": "0",
        "input" : "Are there any restrictions or waiting periods for pre-existing conditions under my health insurance plan?", "output": "0",
        "input" : "Can you clarify the concept of co-payments and coinsurance in relation to health insurance coverage?", "output": "0",
        "input" : "What is the significance of network providers in a health insurance plan, and how do I find out if my preferred doctors are in-network?", "output": "0"
        "input" : "What types of medical services are typically covered under a standard health insurance policy?", "output": "0",
        "input" : "How does the process of filing a health insurance claim work?", "output": "0",
        "input" : "Can you explain the difference between an HMO and a PPO health insurance plan?", "output": "0",
        "input" : "How long does someone have after losing their Medicaid coverage to enroll in a Medicare Advantage plan?", "output": "0",
        "input" : "I've been hearing a lot about the Mediterranean diet and its health benefits. Can you elaborate on what it entails and why it's considered one of the healthiest diets?", "output": "0",
		"input" : "For my upcoming presentation on climate change, I need a brief overview of its effects on polar regions, accompanied by a compelling visual that highlights the melting of ice caps.", "output": "0",
		"input" : "I'm planning a lesson on ancient Egyptian civilization for my history class. Could you give me a succinct summary of their culture, achievements, and an illustration of their architectural marvels, like the pyramids?", "output": "0",
		"input" : "I'm working on a fantasy novel and need some inspiration for the main character's attire. Can you generate an image showing a warrior elf with an intricate armor design, set in an enchanted forest background?", "output": "0",
		"input" : "I'm curious about the architectural blend of traditional Japanese and modern minimalist styles. Could you create a visual concept of a house that incorporates both elements, focusing on harmony and naturalmaterials?", "output": "0",
		"input" : "I've recently taken an interest in quantum computing but find it quite baffling. Could you break down the basics for me, focusing on how quantum bits differ from classical bits and the implications for computing power?", "output": "0",
		"input" : "I'm tasked with developing a wellness program for our employees. Could you outline a comprehensive plan that includes physical activities, mental health strategies, and nutritional advice, tailored to a busy work schedule?", "output": "0",
		"input" : "In preparation for my garden redesign, I need some guidance on creating a cottage garden theme. Please provide a brief description of key elements and plants that are typically used, along with a sketch showing a layout idea.", "output": "0",
		"input" : "I'm doing a project on the Renaissance art movement and its influence on modern art. Could you give me a concise explanation of the main characteristics of Renaissance art and create an image that merges Renaissance style with a contemporary subject?", "output": "0"

        "{text}"

        Dominant Intent Code:
        """

    prompt = PromptTemplate(template=template, input_variables=['text'])

    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)

    result = llm_chain(doc)
    result = result["text"] 
    # print(result)
    
    return int(result.strip('"'))

    
    
# def predict_disease(symptoms):
#     url = "http://localhost:3000/check-symptom"  # Your API endpoint
#     payload = {"symptoms": symptoms}  # Payload to send to the API
#     response = requests.post(url, json=payload)  # Send POST request to the API
#     # response = requests.post(url, json=symptoms)
#     if response.status_code == 200:
#         return response.json()["final_result"]  # Extract predicted disease from the response JSON
#     else:
#         print("Error:", response.text)  # Print error message if request fails
#         return None



def InputProcessor(_text):
    # api_key = "..................................."
    # os.environ['OPENAI_API_KEY'] = api_key
    llm = ChatOpenAI(model_name='gpt-4', temperature=0)
    
    template = """
        Whenever you provide a text input, it could contain a generic sentence mentioning various symptoms
        or just one symptom. If the input text contains symptoms from the following list,
        I'll return a list of the symptoms mentioned. only the list is required as output no other text, just the list.
        
        symptoms_list = ['itching', 'skin rash', 'nodal skin eruptions', 'continuous sneezing', 'shivering',
        'chills', 'joint pain', 'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting', 'vomiting',
        'burning micturition', 'spotting urination', 'fatigue', 'weight gain', 'anxiety', 'cold hands and feets',
        'mood swings', 'weight loss', 'restlessness', 'lethargy', 'patches in throat', 'irregular sugar level',
        'cough', 'high fever', 'sunken eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
        'yellowish skin', 'dark urine', 'nausea', 'loss of appetite', 'pain behind the eyes', 'back pain',
        'constipation', 'abdominal pain', 'diarrhoea', 'mild fever', 'yellow urine', 'yellowing of eyes',
        'acute liver failure', 'fluid overload', 'swelling of stomach', 'swelled lymph nodes', 'malaise',
        'blurred and distorted vision', 'phlegm', 'throat irritation', 'redness of eyes', 'sinus pressure',
        'runny nose', 'congestion', 'chest pain', 'weakness in limbs', 'fast heart rate', 'pain during bowel movements',
        'pain in anal region', 'bloody stool', 'irritation in anus', 'neck pain', 'dizziness', 'cramps', 'bruising',
        'obesity', 'swollen legs', 'swollen blood vessels', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails',
        'swollen extremeties', 'excessive hunger', 'extra marital contacts', 'drying and tingling lips', 'slurred speech',
        'knee pain', 'hip joint pain', 'muscle weakness', 'stiff neck', 'swelling joints', 'movement stiffness',
        'spinning movements', 'loss of balance', 'unsteadiness', 'weakness of one body side', 'loss of smell',
        'bladder discomfort', 'foul smell of urine', 'continuous feel of urine', 'passage of gases', 'internal itching',
        'toxic look (typhos)', 'depression', 'irritability', 'muscle pain', 'altered sensorium', 'red spots over body',
        'belly pain', 'abnormal menstruation', 'dischromic patches', 'watering from eyes', 'increased appetite', 'polyuria',
        'family history', 'mucoid sputum', 'rusty sputum', 'lack of concentration', 'visual disturbances',
        'receiving blood transfusion', 'receiving unsterile injections', 'coma', 'stomach bleeding', 'distention of abdomen',
        'history of alcohol consumption', 'fluid overload', 'blood in sputum', 'prominent veins on calf', 'palpitations',
        'painful walking', 'pus filled pimples', 'blackheads', 'scurring', 'skin peeling', 'silver like dusting',
        'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose', 'yellow crust ooze']
        
        Given the input text below, create the list of symptoms:
        
        Examples:
        "input" : "My friend is sneezing continuously itching and shivering", "output": ['continuous sneezing', 'itching', 'shivering'],
        "input" : "She is having blackheads skin peeling blister red sore around nose", "output": ['blackheads', 'skin peeling', 'blister', 'red sore around nose']
        "input" : "I am having family history of diabetes and obesity", "output": ['family history', 'obesity']
        "input" : "i an having high fever and back pain also blood in my mucus", "output": ['high fever', 'back pain', 'blood in sputum']
        "input" : "i am suffering from diarrhoea and mild fever and yellow urine and yellowing of eyes and acute liver failure and fluid 
        overload and swelling of stomach and swelled lymph nodes and malaise and blurred and distorted vision and phlegm and throat irritation 
        and redness of eyes and sinus pressure and runny nose and congestion and chest pain and weakness in limbs and fast heart rate and pain 
        during bowel movements and pain in anal region and bloody stool and irritation in anus", "output": ['diarrhoea', 'mild fever', 'yellow urine', 
        'yellowing of eyes', 'acute liver failure', 'fluid overload', 'swelling of stomach', 'swelled lymph nodes', 'malaise', 'blurred and distorted vision', 
        'phlegm', 'throat irritation', 'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'chest pain', 'weakness in limbs', 'fast heart rate', 
        'pain during bowel movements', 'pain in anal region', 'bloody stool', 'irritation in anus']
        "input" : "i am having some issues in my back its paining and also i am having a lot of cought and yes i am having high fever too also my 
        one side of body pains very much", "output": ['back pain', 'cough', 'high fever', 'weakness of one body side']
        
        "{text}"
        """
        
    prompt = PromptTemplate(template=template, input_variables=['text'])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)
    result = llm_chain(_text)
    result = result["text"]
    return result


def textLLM(_text):
    
    # api_key = "..........................."
    # os.environ['OPENAI_API_KEY'] = api_key

    llm = ChatOpenAI(model_name='gpt-4', temperature=0)

    template = """
        You are a friendly and informative chatbot built on the GPT-4 architecture. Your primary role is to engage 
        in general conversations, provide accurate information, and assist users with their queries to the best of 
        your ability. Remember to maintain a polite and helpful demeanor throughout the conversation. Use your 
        extensive training data to generate relevant, coherent, and contextually appropriate responses. 
        Avoid providing personal opinions or speculative information. Your responses should be based on factual 
        content and general knowledge up to your last training cut-off in December 2023. Be mindful of user privacy 
        and do not request, store, or disclose any personal or sensitive information. 
        
        You are a part of a multi-modal model, which can generate image as well as text, the image generation is taken 
        care by the other model, so don't respond something like this "I'm sorry for any confusion, but as a text-based AI, 
        I'm unable to create images or provide real-time visual descriptions."
        
        Let's have a great conversation!

        User: "How are you doing?"

        Chatbot: [Your response based on general knowledge and conversational tone, without accessing real-time data]

        "{text}"
        
        """

    prompt = PromptTemplate(template=template, input_variables=['text'])

    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)

    result = llm_chain(_text)
    result = result["text"] 
    
    return result

# model = joblib.load('C:/Users/User/Downloads/prediction_app/main_app/model.pkl')

def check_symptom(symptoms_list, model_path):
        symptoms = ['itching', 'skin rash', 'nodal skin eruptions', 'continuous sneezing', 'shivering', 'chills', 'joint pain', 'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting', 'vomiting', 'burning micturition', 'spotting urination', 'fatigue', 'weight gain', 'anxiety', 'cold hands and feets', 'mood swings', 'weight loss', 'restlessness', 'lethargy', 'patches in throat', 'irregular sugar level', 'cough', 'high fever', 'sunken eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish skin', 'dark urine', 'nausea', 'loss of appetite', 'pain behind the eyes', 'back pain', 'constipation', 'abdominal pain', 'diarrhoea', 'mild fever', 'yellow urine', 'yellowing of eyes', 'acute liver failure', 'fluid overload', 'swelling of stomach', 'swelled lymph nodes', 'malaise', 'blurred and distorted vision', 'phlegm', 'throat irritation', 'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'chest pain', 'weakness in limbs', 'fast heart rate', 'pain during bowel movements', 'pain in anal region', 'bloody stool', 'irritation in anus', 'neck pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen legs', 'swollen blood vessels', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 'swollen extremeties', 'excessive hunger', 'extra marital contacts', 'drying and tingling lips', 'slurred speech', 'knee pain', 'hip joint pain', 'muscle weakness', 'stiff neck', 'swelling joints', 'movement stiffness', 'spinning movements', 'loss of balance', 'unsteadiness', 'weakness of one body side', 'loss of smell', 'bladder discomfort', 'foul smell of urine', 'continuous feel of urine', 'passage of gases', 'internal itching', 'toxic look (typhos)', 'depression', 'irritability', 'muscle pain', 'altered sensorium', 'red spots over body', 'belly pain', 'abnormal menstruation', 'dischromic patches', 'watering from eyes', 'increased appetite', 'polyuria', 'family history', 'mucoid sputum', 'rusty sputum', 'lack of concentration', 'visual disturbances', 'receiving blood transfusion', 'receiving unsterile injections', 'coma', 'stomach bleeding', 'distention of abdomen', 'history of alcohol consumption', 'fluid overload', 'blood in sputum', 'prominent veins on calf', 'palpitations', 'painful walking', 'pus filled pimples', 'blackheads', 'scurring', 'skin peeling', 'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose', 'yellow crust ooze']
        disease = {"1": "Fungal infection", "2": "Allergy", "3": "GERD", "4": "Chronic cholestasis", "5": "Drug Reaction", "6": "Peptic ulcer diseae", "7": "AIDS", "8": "Diabetes", "9": "Gastroenteritis", "10": "Bronchial Asthma", "11": "Hypertension", "12": "Migraine", "13": "Cervical spondylosis", "14": "Paralysis (brain hemorrhage)", "15": "Jaundice", "16": "Malaria", "17": "Chicken pox", "18": "Dengue", "19": "Typhoid", "20": "hepatitis A", "21": "Hepatitis B", "22": "Hepatitis C", "23": "Hepatitis D", "24": "Hepatitis E", "25": "Alcoholic hepatitis", "26": "Tuberculosis", "27": "Common Cold", "28": "Pneumonia", "29": "Dimorphic hemmorhoids(piles)", "30": "Heart attack", "31": "Varicose veins", "32": "Hypothyroidism", "33": "Hyperthyroidism", "34": "Hypoglycemia", "35": "Osteoarthristis", "36": "Arthritis", "37": "(vertigo) Paroymsal  Positional Vertigo", "38": "Acne", "39": "Urinary tract infection", "40": "Psoriasis", "41": "Impetigo"}
        intersection_ary = set(symptoms).intersection(symptoms_list)
        comparison_result = [1 if x in intersection_ary else 0 for x in symptoms]
        comparison_result_2d = np.array(comparison_result).reshape(1, -1)
        model = joblib.load(model_path)
        predictions = model.predict(comparison_result_2d)[0]
        final_result = disease.get(str(predictions),'not found')

        return final_result


def MLProcessing(text_input, question=None, model="app/router/model.pkl"):
    # if value==1:
    # age = input("Please enter your age: ")
    # print("You entered:", age)
    
    # symptoms = input("Please enter a list of symptoms separated by commas: ")
    if question is True:
        text_data = "Please tell all the health issues you may be experiencing."
        return text_data
    
    if question is False:
        symptoms_list = InputProcessor(text_input)
        import ast
        symptoms_list_ = ast.literal_eval(symptoms_list)

        # Printing the actual list
        # print(actual_list)
        # symptoms = text_input
        # symptoms_list = symptoms.split(' ')
        print("symptoms_list", symptoms_list_)
        # Predict the disease using the ML model API
        # predicted_disease = predict_disease(symptoms_list_)
        predicted_disease = check_symptom(symptoms_list_, model)
        
        # Print the predicted disease
        # if predicted_disease:
            # print("Based on the symptoms provided, the predicted disease is:", predicted_disease)
        
        insurance_plans = {"Fungal infection": "FungalGuard Insurance", "Allergy": "AllerShield Insurance", "GERD": "GERD Care Insurance", 
                        "Chronic cholestasis": "CholestaSure Insurance", "Drug Reaction": "DrugGuard Insurance", "Peptic ulcer diseae": "UlcerShield Insurance", 
                        "AIDS": "AIDS Assurance", "Diabetes": "DiaCare Insurance", "Gastroenteritis": "GastroGuard Insurance", 
                        "Bronchial Asthma": "AsthmaShield Insurance", "Hypertension": "HyperTensionCare Insurance", 
                        "Migraine": "Migraine Relief Insurance", "Cervical spondylosis": "CervicalEase Insurance", 
                        "Paralysis (brain hemorrhage)": "Paralysis Protection Plan", "Jaundice": "Jaundice Shield Insurance", 
                        "Malaria": "MalariaSafe Insurance", "Chicken pox": "ChickenPoxGuard Insurance", "Dengue": "DengueDefender Insurance", 
                        "Typhoid": "TyphoidCare Insurance", "hepatitis A": "Hepatitis Shield Insurance", "Hepatitis B": "Hepatitis Shield Insurance", 
                        "Hepatitis C": "Hepatitis Shield Insurance", "Hepatitis D": "Hepatitis Shield Insurance", "Hepatitis E": "Hepatitis Shield Insurance", 
                        "Alcoholic hepatitis": "Alcoholic Hepatitis Coverage", "Tuberculosis": "TBGuard Insurance", "Common Cold": "Cold & Flu Coverage", 
                        "Pneumonia": "PneumoShield Insurance", "Dimorphic hemmorhoids(piles)": "Hemorrhoid Relief Insurance", 
                        "Heart attack": "HeartAttackGuard Insurance", "Varicose veins": "VeinVitality Insurance", "Hypothyroidism": "ThyroidCare Insurance", 
                        "Hyperthyroidism": "ThyroidCare Insurance", "Hypoglycemia": "GlucoseGuard Insurance", "Osteoarthristis": "OsteoEase Insurance", 
                        "Arthritis": "Arthritis Aid Insurance", "(vertigo) Paroymsal  Positional Vertigo": "Vertigo Relief Insurance", "Acne": "AcneGuard Insurance", 
                        "Urinary tract infection": "UTI Protection Plan", "Psoriasis": "Psoriasis Shield Insurance", "Impetigo": "Impetigo Coverage"}
        
        if predicted_disease in insurance_plans:
            # print("Insurance Plan for", predicted_disease, ":", insurance_plans[predicted_disease])
            text_data = f"Insurance Plan for {predicted_disease} is {insurance_plans[predicted_disease]}"
            return text_data
            
        else:
            # print("No insurance plan found for", predicted_disease)
            text_data = f"No insurance plan found for {predicted_disease}"
            return text_data
            
        


def router(text_input):
    decision = mlClassifier(text_input)
    if decision==1:
        print("ML Prediction")
        MLProcessing()
    elif decision==0:
        print("LLM Generation")
        _textLLM = textLLM(text_input)
        print(_textLLM)


def main():
    
    os.environ["OPENAI_API_KEY"] = "..................."
    os.environ["AUTOGEN_USE_DOCKER"] = "0"
    
    router("recommend a health insurance")

    # value = mlClassifier(text_input)
    # MLProcessing(value)

if __name__== "__main__":
    main()
    
    