##########the_line_safety_chatbot.py########

'''

cd /code/
streamlit run the_line_safety_chatbot.py --server.port 3412 –-server.address=0.0.0.0

localhost:3412

https://github.com/jiaaro/pydub/blob/master/API.markdown

'''

import os
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import html
#from audiorecorder import audiorecorder
from google_doc import *

st.set_page_config(
	page_title='The Line', 
	layout = 'centered', 
	page_icon = 'logo.png', 
	initial_sidebar_state = 'collapsed')

st.title("The Line Safety ChatBot")

def on_btn_click():
	del st.session_state.messages[:]

def on_btn_reload_qa():

	## download the excel
	download_file_from_google_drive(
		id = '1HPUapW8V_3FWP6uvxrfRy5A7yzr1VIO6', 
		destination = 'hr_training_pairs.xlsx',
		)

	## read the qa pairs
	hr_training_pairs = pd.read_excel(
		'hr_training_pairs.xlsx',
		)
	hr_training_pairs = hr_training_pairs[['Question', 'Answer']]
	hr_training_pairs = hr_training_pairs.drop_duplicates()
	hr_training_pairs = hr_training_pairs.to_dict('records')
	print(f"loaded {len(hr_training_pairs)} qa pairs. embedding the qa pairs")

	## embedding
	st.session_state['qa_pairs'] = []

	start_time = time.time()
	for r in hr_training_pairs:
		try:

			Question_embedding = requests.post(
				'http://37.224.68.132:27329/text_embedding/all_mpnet_base_v2',
				json = {
				  "text": r['Question']
				}
				).json()['embedding_vector']

			Answer_embedding = requests.post(
				'http://37.224.68.132:27329/text_embedding/all_mpnet_base_v2',
				json = {
				  "text": r['Answer']
				}
				).json()['embedding_vector']

			st.session_state['qa_pairs'].append({
				'Question':r['Question'],
				'Answer':r['Answer'],
				'Question_embedding':Question_embedding,
				'Answer_embedding':Answer_embedding,
				})

		except:
			print(f"{r['Question']}, {r['Answer']}")

	pd.DataFrame(st.session_state['qa_pairs']).to_json('qa_pairs_embeddings.json', lines = True, orient = 'records',)

	print(f"ebmedding of {len(st.session_state['qa_pairs'])} qa pairs complete. runing time {time.time() - start_time: 0.2f} s")


if 'messages' not in st.session_state:
	st.session_state['messages'] = [
	{
	"role":"assistant",
	"content":"""Welcome to The Line Safety Chatbot, feel free to ask me any general questions regarding construction safety, such as: "What should be ensured before work commences?" """,
	}
	]

if 'qa_pairs' not in st.session_state:
	st.session_state['qa_pairs'] = pd.read_json('qa_pairs_embeddings.json', lines = True, orient = 'records',).to_dict('records')

system_prompt = f'You are a large language model named The Line Safety Chatbot developed by TONOMUS to answer general questions regarding construction safety, such as "What should be ensured before work commences?". Only respond to the last instruction. Your response should be short and abstract, less than 64 words. Do not try to continue the conversation by generating more instructions. Stop generating more responses when the current generated response is longer than 64 tokens. Conversations should flow, and be designed in a way to not reach a dead end by ending responses with "Do you have any further questions?"'


def on_input_change():
	user_input = st.session_state.user_input
	if not len(user_input) > 0:
		return None

	# show the user input
	st.session_state['messages'].append({
		"role":"user",
		"content":user_input,
		})

	# embedding of the input
	input_embedding = requests.post(
		'http://37.224.68.132:27329/text_embedding/all_mpnet_base_v2',
		json = {
		"text": user_input
		}
		).json()['embedding_vector']

	# score the qa pairs
	similar_qas = []

	for r in st.session_state['qa_pairs']:	

		question_score = np.dot(
			np.array(input_embedding),
			np.array(r['Question_embedding']),
			)

		# if the question matches the qa, return the answer
		if question_score >= 0.9:
			st.session_state['messages'].append({
				"role":"assistant",
				"content":r['Answer'],
				})
			return 

		answer_score = np.dot(
			np.array(input_embedding),
			np.array(r['Answer_embedding']),
			)

		overall_score = np.max([question_score,answer_score])
		if overall_score >= 0.5:
			similar_qas.append({
				'Question':r['Question'],
				'Answer':r['Answer'],
				'question_score':question_score,
				'answer_score':answer_score,
				'overall_score':overall_score
				})

	similar_qas = sorted(similar_qas, key=lambda x: x['overall_score'],)

	# prompt engineering

	prompt_conversation = []

	for r in similar_qas[-4:]:
		prompt_conversation.append(f"[INST] {r['Question'].strip()} [/INST]")
		prompt_conversation.append(f"{r['Answer'].strip()}")

	for m in st.session_state['messages'][-4:]:
		if m['role'] == 'user':
			prompt_conversation.append(f"[INST] {m['content'].strip()} [/INST]")
		else:
			prompt_conversation.append(f"{m['content'].strip()}")

	prompt_conversation = ',\n'.join(prompt_conversation)


	prompt = f"""
<<SYS>> {system_prompt} <</SYS>>

{prompt_conversation}
	"""

	#print(prompt)

	response = requests.post(
		'http://37.224.68.132:27427/llama_2_7b_chat_gptq/generate',
		json = {'prompt':prompt}
		)
	response = response.json()['response']

	response = response.split('[INST')[0].strip()

	st.session_state['messages'].append({
		"role":"assistant",
		"content":response,
		})



chat_placeholder = st.empty()

with chat_placeholder.container():  
	st.button("Clear message", on_click=on_btn_click)
	st.button("Reload the QA pairs", on_click=on_btn_reload_qa)
	i = 0
	for m in st.session_state['messages']:
		try:
			if m['role'] == 'user':
				is_user=True
			else:
				is_user=False

			message(
			m['content'], 
			key=f"{i}", 
			allow_html=True,
			is_user = is_user,
			is_table=False,
			)
		except:
			pass
		i+=1


with st.container():

	text = ""

	st.text_input("User Input:", value = text, key="user_input")
	st.button('Send', on_click = on_input_change)


##########the_line_safety_chatbot.py########