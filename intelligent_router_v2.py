import io
import json
import os
import pdb
import random
import re
import time
import sys
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import uuid
import matplotlib.pyplot as plt
import PIL
import requests
from diskcache import Cache
from openai import OpenAI
from termcolor import colored
import pandas as pd
import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.img_utils import _to_pil, get_image_data
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from app.core.S3Storage import S3Storage
from app.router.ml_router import mlClassifier, MLProcessing


s3storage = S3Storage()
def dalle_call(client: OpenAI, model: str, prompt: str, size: str, quality: str, n: int) -> str:
	"""
	Generate an image using OpenAI's DALL-E model and cache the result.

	This function takes a prompt and other parameters to generate an image using OpenAI's DALL-E model.
	It checks if the result is already cached; if so, it returns the cached image data. Otherwise,
	it calls the DALL-E API to generate the image, stores the result in the cache, and then returns it.

	Args:
		client (OpenAI): The OpenAI client instance for making API calls.
		model (str): The specific DALL-E model to use for image generation.
		prompt (str): The text prompt based on which the image is generated.
		size (str): The size specification of the image. TODO: This should allow specifying landscape, square, or portrait modes.
		quality (str): The quality setting for the image generation.
		n (int): The number of images to generate.

	Returns:
	str: The image data as a string, either retrieved from the cache or newly generated.

	Note:
	- The cache is stored in a directory named '.cache/'.
	- The function uses a tuple of (model, prompt, size, quality, n) as the key for caching.
	- The image data is obtained by making a secondary request to the URL provided by the DALL-E API response.
	"""
	# Function implementation...
	cache = Cache(".cache/")  # Create a cache directory
	key = (model, prompt, size, quality, n)
	if key in cache:
		return cache[key]

	# If not in cache, compute and store the result
	response = client.images.generate(
		model=model,
		prompt=prompt,
		size=size,
		quality=quality,
		n=n,
	)
	image_url = response.data[0].url
	img_data = get_image_data(image_url)
	cache[key] = img_data

	return img_data


def extract_img(agent: Agent) -> PIL.Image:
	"""
	Extracts an image from the last message of an agent and converts it to a PIL image.

	This function searches the last message sent by the given agent for an image tag,
	extracts the image data, and then converts this data into a PIL (Python Imaging Library) image object.

	Parameters:
		agent (Agent): An instance of an agent from which the last message will be retrieved.

	Returns:
		PIL.Image: A PIL image object created from the extracted image data.

	Note:
	- The function assumes that the last message contains an <img> tag with image data.
	- The image data is extracted using a regular expression that searches for <img> tags.
	- It's important that the agent's last message contains properly formatted image data for successful extraction.
	- The `_to_pil` function is used to convert the extracted image data into a PIL image.
	- If no <img> tag is found, or if the image data is not correctly formatted, the function may raise an error.
	"""
	# Function implementation...
	img_data = re.findall("<img (.*)>", agent.last_message()["content"])[0]
	pil_img = _to_pil(img_data)
	return pil_img



class DALLEAgent(ConversableAgent):
	def __init__(self, name, llm_config: dict, **kwargs):
		super().__init__(name, llm_config=llm_config, **kwargs)

		# try:
		# 	config_list = llm_config["config_list"]
		# 	api_key = config_list[0]["api_key"]
		# except Exception as e:
		# 	print("Unable to fetch API Key, because", e)
		api_key = os.getenv("OPENAI_API_KEY")
		self.client = OpenAI(api_key=api_key)
		self.register_reply([Agent, None], DALLEAgent.generate_dalle_reply)

	def send(
		self,
		message: Union[Dict, str],
		recipient: Agent,
		request_reply: Optional[bool] = None,
		silent: Optional[bool] = False,
	):
		# override and always "silent" the send out message;
		# otherwise, the print log would be super long!
		super().send(message, recipient, request_reply, silent=True)

	def generate_dalle_reply(self, messages: Optional[List[Dict]], sender: "Agent", config):
		"""Generate a reply using OpenAI DALLE call."""
		client = self.client if config is None else config
		if client is None:
			return False, None
		if messages is None:
			messages = self._oai_messages[sender]

		prompt = messages[-1]["content"]
		# TODO: integrate with autogen.oai. For instance, with caching for the API call
		img_data = dalle_call(
			client=self.client,
			model="dall-e-3",
			prompt=prompt,
			size="1024x1024",  # TODO: the size should be flexible, deciding landscape, square, or portrait mode.
			quality="standard",
			n=1,
		)
		out_message = f"<img {img_data}>"
		return True, out_message



def text_image_classifier(doc):
	
	llm = ChatOpenAI(model_name='gpt-4', temperature=0)

	template = """
		As a classifier, your task is to analyze text inputs and determine their dominant intent based on whether 
		they refer to a request or information related to images, text, or both. When inputs contain multiple 
		elements, focus on the final action or the most dominant element of the request. Assign a numerical 
		code based on these categories:

		- 0: Image (or creating an image)
		- 1: Text (purely informational or textual content)
		- 2: Both Image and Text (requests that involve both creating an image and providing textual information)

		Given the input text below, classify it according to the dominant intent or final action requested:

		Examples:
		"input": "Create an image of a sunset", "output": "0",
		"input": "Describe a sunset", "output": "1",
		"input": "Describe a sunset and create an image of it", "output": "2"
		"input": "create image of the first colour in rainbow", "output": "0",
		"input": "dog eating hotdog with wales in it, write about it", "output": "1",
		"input": "i want to play cricket with dhoni. And i need to take a photo with him", "output": "1",
		"input": "i wanted to meet sachin and i have a photo of him. can you suggest me the way to reach him? can 
		you a draw a image of him", "output": "2",
		"input": "I'm curious about what a fusion between Art Nouveau style and cyberpunk elements would look like 
		in a cityscape. Can you visualize that for me?", "output": "0",
		"input": "I've always been fascinated by the concept of an underwater restaurant. Could you create an image 
		that depicts what dining under the sea might look like, with marine life visible through large glass windows?", "output": "0",
		"input": "I'm writing an article on the impact of blockchain technology on the finance sector. Could you 
		provide me with a detailed analysis of its current applications and future potential?", "output": "1",
		"input": "I've been hearing a lot about the Mediterranean diet and its health benefits. Can you elaborate on 
		what it entails and why it's considered one of the healthiest diets?", "output": "1",
		"input": "For my upcoming presentation on climate change, I need a brief overview of its effects on polar 
		regions, accompanied by a compelling visual that highlights the melting of ice caps.", "output": "2",
		"input": "I'm planning a lesson on ancient Egyptian civilization for my history class. Could you give me a 
		succinct summary of their culture, achievements, and an illustration of their architectural marvels, like the 
		pyramids?", "output": "2",
		"input": "I'm working on a fantasy novel and need some inspiration for the main character's attire. Can you 
		generate an image showing a warrior elf with an intricate armor design, set in an enchanted forest background?", "output": "0",
		"input": "I'm curious about the architectural blend of traditional Japanese and modern minimalist styles. 
		Could you create a visual concept of a house that incorporates both elements, focusing on harmony and natural
		  materials?", "output": "0",
		"input": "I've recently taken an interest in quantum computing but find it quite baffling. Could you break 
		down the basics for me, focusing on how quantum bits differ from classical bits and the implications for 
		computing power?", "output": "1",
		"input": "I'm tasked with developing a wellness program for our employees. Could you outline a comprehensive 
		plan that includes physical activities, mental health strategies, and nutritional advice, tailored to a 
		busy work schedule?", "output": "1",
		"input": "In preparation for my garden redesign, I need some guidance on creating a cottage garden theme. 
		Please provide a brief description of key elements and plants that are typically used, along with a sketch showing a layout idea.", "output": "2",
		"input": "I'm doing a project on the Renaissance art movement and its influence on modern art. Could you 
		give me a concise explanation of the main characteristics of Renaissance art and create an image that 
		merges Renaissance style with a contemporary subject?", "output": "2"
		
		
		"{text}"

		Dominant Intent Code:
		"""

	prompt = PromptTemplate(template=template, input_variables=['text'])

	llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)

	result = llm_chain(doc)
	result = result["text"] 
	
	return int(result.strip('"'))


def textLLM(_text, template_content):

	llm = ChatOpenAI(model_name='gpt-4', temperature=0)

	# template = """
	# 	You are a friendly and informative chatbot built on the GPT-4 architecture. Your primary role is to engage 
	# 	in general conversations, provide accurate information, and assist users with their queries to the best of 
	# 	your ability. Remember to maintain a polite and helpful demeanor throughout the conversation. Use your 
	# 	extensive training data to generate relevant, coherent, and contextually appropriate responses. 
	# 	Avoid providing personal opinions or speculative information. Your responses should be based on factual 
	# 	content and general knowledge up to your last training cut-off in December 2023. Be mindful of user privacy 
	# 	and do not request, store, or disclose any personal or sensitive information. 
		
	# 	You are a part of a multi-modal model, which can generate image as well as text, the image generation is taken 
	# 	care by the other model, so don't respond something like this "I'm sorry for any confusion, but as a text-based AI, 
	# 	I'm unable to create images or provide real-time visual descriptions."
		
	# 	Let's have a great conversation!

	# 	User: "How are you doing?"

	# 	Chatbot: [Your response based on general knowledge and conversational tone, without accessing real-time data]

	# 	"{text}"
		
	# 	"""
  
	template = f"""
        {template_content}
        
        "{{text}}"
    """

	prompt = PromptTemplate(template=template, input_variables=['text'])

	llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)

	result = llm_chain(_text)
	result = result["text"] 
	
	return result


def textLLM_(user, template_content):
    try:
        model_kwargs = {
            "max_new_tokens": 4096,
            "temperature": 0.1,
            "repetition_penalty": 1.176,
            "top_k": 40,
            "top_p": 0.1
        }
        
        api_url = "........................"
        
        inputs = {
            "inputs": f"{template_content}|{user}|:",
            "parameters": model_kwargs
        }
        
        response = requests.post(api_url, json=inputs)
        
        if response.status_code == 200:
            text = response.json()
            message = text['message'][0]
            return message['generated_text']
            
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



class Intelligent_Router():

	def __init__(self, template_content, csv = "app/router/initial_data.csv"):

		# self.config_list_gpt4 = autogen.config_list_from_json(
		# 	"OAI_CONFIG_LIST",
		# 	filter_dict={
		# 		"model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
		# 	},
		# )

		# self.config_list_dalle = autogen.config_list_from_json(
		# 					env_or_file="OAI_CONFIG_LIST",
		# 					file_location=".",
		# 					filter_dict={
		# 						"model": {
		# 							"Dalle",
		# 						}
		# 					}
		# 				)
		# print(self.config_list_dalle)
		# self.gpt4_llm_config = {"config_list": self.config_list_gpt4, "cache_seed": 42}

		# os.environ["OPENAI_API_KEY"] = "..............................."
		# os.environ["AUTOGEN_USE_DOCKER"] = "0"


		# self.dalle = DALLEAgent(name="Dalle", llm_config={"config_list": self.config_list_dalle})
		self.dalle = DALLEAgent(name="Dalle", llm_config={})

		self.user_proxy = UserProxyAgent(
			name="User_proxy", system_message="A human admin.", human_input_mode="NEVER", max_consecutive_auto_reply=0
			
		)

		# self.csv_file = pd.read_csv(csv)
		# data = {
		# 	'user_input': [None],
		# 	'model_response': [None],
		# 	'model_used': [None]
		# }

		# # Create a DataFrame from the sample data
		# df = pd.DataFrame(data)

		# # Save the DataFrame to a CSV file
		self.csv_file_path = csv
		# df.to_csv(self.csv_file_path, index=False)
		self.csv_file = pd.read_csv(self.csv_file_path)
		self.template_content = template_content
		self.text = None
		self.image = None
		self.ML = None

	def text_image_router(self, text_input):

		decision = text_image_classifier(text_input)

		if decision==0:
			self.image = True
			# return self.image

		elif decision==1:
			self.text = True
			# return self.text

		elif decision==2:
			self.image = True
			self.text = True
			# return 

	def ml_router(self, text_input):
		decision = mlClassifier(text_input)
		if decision==1:
			self.ML = True
		elif decision==0:
			self.ML = False
			
   
	def aggregator(self, text_input):

		text_data = None
		image_data = None
		question = None
  
		csv_file_last = self.csv_file.iloc[-1]["model_used"]
		csv_file_last_r = self.csv_file.iloc[-1]["model_response"]
		csv_file_s_last = self.csv_file.iloc[-2]["model_used"]

		print("csv_file_last", csv_file_last)
		print("csv_file_s_last", csv_file_s_last)
		# print("csv_file_last", csv_file_last)

		
		if csv_file_last=="ML" and csv_file_s_last!="ML":
			question = False
			text_data = MLProcessing(text_input, question)
			text_data = str(text_data)
			print(text_data)
			new_row = {'user_input': text_input, 'model_response': text_data, 'model_used': 'ML'}
			# Convert the new_row dictionary to a DataFrame
			new_row_df = pd.DataFrame([new_row])
			# Append the new row DataFrame to the existing DataFrame
			self.csv_file = pd.concat([self.csv_file, new_row_df], ignore_index=True)
			# Save the updated DataFrame back to the same CSV file
			self.csv_file.to_csv(self.csv_file_path, index=False)
			self.ML = None
			return {"image": image_data, "text": text_data}
		elif csv_file_last=="ML" and csv_file_last_r=="Please tell all the health issues you may be experiencing.":
			question = False
			text_data = MLProcessing(text_input, question)
			text_data = str(text_data)
			print(text_data)
			new_row = {'user_input': text_input, 'model_response': text_data, 'model_used': 'ML'}
			# Convert the new_row dictionary to a DataFrame
			new_row_df = pd.DataFrame([new_row])
			# Append the new row DataFrame to the existing DataFrame
			self.csv_file = pd.concat([self.csv_file, new_row_df], ignore_index=True)
			# Save the updated DataFrame back to the same CSV file
			self.csv_file.to_csv(self.csv_file_path, index=False)
			self.ML = None
			return {"image": image_data, "text": text_data}
		else:
			question = True
   
		self.ml_router(text_input)
		self.text_image_router(text_input)

		# self.ML = False #just for testing

		if self.ML is True:
			self.text = False
		print("text", self.text)
		print("image", self.image)
		print("ML", self.ML)
  
		if self.ML is True:
			print("ML Prediction")
			text_data = MLProcessing(text_input, question)
			text_data = str(text_data)
			print(text_data)
			new_row = {'user_input': text_input, 'model_response': text_data, 'model_used': 'ML'}
			# Convert the new_row dictionary to a DataFrame
			new_row_df = pd.DataFrame([new_row])
			# Append the new row DataFrame to the existing DataFrame
			self.csv_file = pd.concat([self.csv_file, new_row_df], ignore_index=True)
			self.csv_file.to_csv(self.csv_file_path, index=False)
			self.ML = None
   
		if self.text is True and self.image is True:
			print("I am in >> both text and image")
			_textLLM = textLLM(text_input, self.template_content)
			text = f"{text_input} {_textLLM}"
			self.user_proxy.initiate_chat(self.dalle, message=text)
			img = extract_img(self.dalle)
			uuid_str = str(uuid.uuid4()).replace("-", "")
			image_id = f"image_{uuid_str}.png"
			img_buffer = io.BytesIO()
			# Save the image to the buffer in PNG format
			img.save(img_buffer, format='PNG')
			# Move to the beginning of the buffer
			img_buffer.seek(0)

			s3storage.upload_image(s3storage.bucket_name, img_buffer, f"images/{image_id}")
			# s3storage.save_file(s3storage.bucket_name, image_data, image_id)
			# get url
			image_url = s3storage.get_object_url(s3storage.bucket_name, f"images/{image_id}")
			image_data = image_url
			text_data = text

		elif self.text is True:
			print("I am in >> text")
			text = textLLM(text_input, self.template_content)
			self.text = None
			text_data = text
			new_row = {'user_input': text_input, 'model_response': text_data, 'model_used': 'text'}
			# Convert the new_row dictionary to a DataFrame
			new_row_df = pd.DataFrame([new_row])
			# Append the new row DataFrame to the existing DataFrame
			self.csv_file = pd.concat([self.csv_file, new_row_df], ignore_index=True)
			# Save the updated DataFrame back to the same CSV file
			self.csv_file.to_csv(self.csv_file_path, index=False)

		elif self.image is True:
			print("I am in >> image")
			self.user_proxy.initiate_chat(self.dalle, message=text_input)
			img = extract_img(self.dalle)
			uuid_str = str(uuid.uuid4()).replace("-", "")
			image_id = f"image_{uuid_str}.png"
			
			# print(f"Image saved as {image_dir}")
			# self.image = None
			# image_data = image_id
			img_buffer = io.BytesIO()
			# Save the image to the buffer in PNG format
			img.save(img_buffer, format='PNG')
			# Move to the beginning of the buffer
			img_buffer.seek(0)

			s3storage.upload_image(s3storage.bucket_name, img_buffer, f"images/{image_id}")
			# s3storage.save_file(s3storage.bucket_name, image_data, image_id)
			# get url
			image_url = s3storage.get_object_url(s3storage.bucket_name, f"images/{image_id}")
			# remove local file
			# os.remove(image_dir)
			image_data = image_url

		return {"image": image_data, "text": text_data}
	

def main():
    
	# template_content = """
	#  	You are a friendly and informative chatbot built on the GPT-4 architecture. Your primary role is to engage 
	# 	in general conversations, provide accurate information, and assist users with their queries to the best of 
	# 	your ability. Remember to maintain a polite and helpful demeanor throughout the conversation. Use your 
	# 	extensive training data to generate relevant, coherent, and contextually appropriate responses. 
	# 	Avoid providing personal opinions or speculative information. Your responses should be based on factual 
	# 	content and general knowledge up to your last training cut-off in December 2023. Be mindful of user privacy 
	# 	and do not request, store, or disclose any personal or sensitive information. 
		
	# 	You are a part of a multi-modal model, which can generate image as well as text, the image generation is taken 
	# 	care by the other model, so don't respond something like this "I'm sorry for any confusion, but as a text-based AI, 
	# 	I'm unable to create images or provide real-time visual descriptions."
		
	# 	Let's have a great conversation!

	# 	User: "How are you doing?"

	# 	Chatbot: [Your response based on general knowledge and conversational tone, without accessing real-time data]
	# """
	template_content = "Your task is to answer questions about Insurance policies and rules and regulations. You will answer questions and not generate anything on its own."
 
	# csv = 'initial_data.csv'
	intelligent_router = Intelligent_Router(template_content)
	# text_input = sys.argv[1]
	text_input = "i am having some issues in my back its paining and also i am having a lot of cought and yes i am having high fever too also my one side of body pains very much"
	data = intelligent_router.aggregator(text_input)
	print(data)
	
	
if __name__=="__main__":
	main()
 
# "recommend me some health insurance plan"
