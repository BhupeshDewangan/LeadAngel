import requests
import bs4
from bs4 import BeautifulSoup
import re

import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

# API KEY
import google.generativeai as genai

# Access the secret using userdata.get()
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel(model_name="gemini-1.5-flash")


def soup(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    return soup

def remove_html_script_style_tags(html: str) -> str:
    # Remove script and style tags with their content
    html = re.sub(r'<script.*?>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style.*?>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove head and its content if needed
    # html = re.sub(r'<head.*?>.*?</head>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Optionally remove other tags like title, meta, link
    # html = re.sub(r'<(meta|link).*?>', '', html, flags=re.IGNORECASE)

    # Remove all other remaining HTML tags
    html = re.sub(r'<[^>]+>', '', html)

    # Remove extra whitespace and blank lines
    text = re.sub(r'\s+', ' ', html)
    return text.strip()


def get_data(clean_text, question_ans):
    """
    Create synthetic data for creating the PowerPoints in JSON format.

  Args:
    The number of PowerPoint files
    The subject for these files
    """

    # prompt = f"""read this only informative content {clean_text} 
    # and i am going to ask some questions from you, you have to give ans to me of {question_ans} """

    prompt = f"""You are a helpful and respectful sales chat assistant who answers queries relevant only to the context/provied text. 
    Please answer all questions politely. Use a conversational tone, like you're chatting with someone, not like you're writing an email. 
    If the user asks about anything outside of the sales data like if they ask something irrelevant, simply say, 'I can only provide answers related to the relevent text, sir.
    Here is the context: {clean_text}  
    Now, please answer the following question: {question_ans} 
    """

    response = model.generate_content(prompt)
    spe_char_to_remove = ['`', '`', 'json']
    output =  response.text
    for character in spe_char_to_remove:
      output = output.replace(character, '')

    return output



# if __name__ == "__main__":
#     st.set_page_config(page_title="Generate Q&A from Website", page_icon="🤖")
#     st.title("Generate Q&A from Website")
    
#     # Initialize session state
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [
#             {"role": "AI", "content": "Hello, I am a bot. How can I help you?"},
#         ]
#     if "show_question_box" not in st.session_state:
#         st.session_state.show_question_box = False
    
#     # Sidebar for URL input and Next button
#     with st.sidebar:
#         st.header("Settings")
#         website_url = st.text_input("Website URL", key="website_url_input")
#         if st.button('Next'):
#             # Clear chat history when URL is updated
#             st.session_state.chat_history = []
#             st.session_state.show_question_box = True
    
#     # Check if URL is provided
#     if not website_url:
#         st.info("Please enter a website URL")
#     else:
#         # When next is clicked and URL is pasted, show question input box
#         if st.session_state.show_question_box:
#             user_query = st.text_input("Type your message here...", key="user_input")

            
#             if user_query:
#                 # Call your functions: soup, remove_html_script_style_tags, get_data
#                 soup_data = soup(website_url)
#                 clean_text = remove_html_script_style_tags(str(soup_data))
#                 response = get_data(clean_text, user_query)
#                 # Update chat history
#                 # st.spinner("Processing...")
#                 st.session_state.chat_history.append({"role": "Human", "content": user_query})
#                 st.session_state.chat_history.append({"role": "AI", "content": response})
#         else:
#             st.info("Click 'Next' in the sidebar to start asking questions.")

#         # Display chat history
#         for message in st.session_state.chat_history:
#             if message["role"] == "AI":
#                 with st.chat_message("AI"):
#                     st.write(message["content"])
#             elif message["role"] == "Human":
#                 with st.chat_message("Human"):
#                     st.write(message["content"])


