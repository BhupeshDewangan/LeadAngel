import streamlit as st
from main import *
import time
# from pinerag import *
from src.app import *


def clear_url_input():
    st.session_state['url_input'] = ""

# Sidebar selectbox
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("AI Assistant", "RAG + Vector DB"),
    key="contact_method",
    on_change=clear_url_input
)


st.sidebar.divider()


def clear_conversation():
        st.session_state.messages = []

def display_chat_history():
        for msg in st.session_state.messages:
            with messages.chat_message(msg["role"]):
                st.write(msg["content"])

if add_selectbox == "AI Assistant":

    # Sidebar input box below selectbox
    url_input = st.sidebar.text_input("Enter URL here:", key="url_input")

    # Sidebar Next button
    next_clicked = st.sidebar.button("Next")

    if next_clicked:
        st.session_state.messages = []  # Clear chat history when Next is clicked
        st.session_state.show_question_box = True

        with st.sidebar:
            with st.status("Downloading data...", expanded=True) as status:
                st.write("Searching for data...")
                time.sleep(2)
                st.write("Found URL.")
                time.sleep(1)
                st.write("Extracting data...")
                time.sleep(1)
                st.write("AI Ready...")
                time.sleep(1)
                status.update(
                    label="Download complete!", state="complete", expanded=False
                )

    # Initialize the session state variable to store messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Container to hold the chat messages
    messages = st.container()

    if url_input:
        soup_data = soup(url_input)
        clean_text = remove_html_script_style_tags(str(soup_data))

    # Accept user input
    user_query = st.chat_input("Type your message here...")
    
    if user_query:

        st.session_state.messages.append({"role": "user", "content": user_query})
        
        response = get_data(clean_text, user_query)
        
        # Generate assistant response (here simply echo for demo)
        assistant_response = f"Echo: {response}"
        
        # Append assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})


    st.button("Clear message", on_click = clear_conversation)
    display_chat_history()


else:
    # st.write("RAG + Vector DB Selected")

    url_input = st.sidebar.text_input("Enter URL here:", key="url_input")

    # Sidebar Next button
    next_clicked = st.sidebar.button("Next")

    if next_clicked:
        st.session_state.messages = []  # Clear chat history when Next is clicked
        st.session_state.show_question_box = True


        

    # Initialize the session state variable to store messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Container to hold the chat messages
    messages = st.container()


    if url_input:
        soup_data = soup(url_input)
        clean_text = remove_html_script_style_tags(str(soup_data))
        chunks = split_documents_into_chunks(clean_text)
        index = upsert_to_pinecone(chunks)

        with st.sidebar:
            with st.status("Downloading data...", expanded=True) as status:
                st.write("Searching for data...")
                time.sleep(2)
                st.write("Found URL.")
                time.sleep(1)
                st.write("Extracting Data...")
                time.sleep(2)
                st.write("Chunking...")
                time.sleep(1)
                st.write("Chunking... Embedding...and Uploading into Vector DB...")
                time.sleep(1)
                st.write("RAG + Vector DB AI Ready")
                time.sleep(1)
                status.update(
                    label="Download complete!", state="complete", expanded=False
                )

    # Accept user input
    user_query = st.chat_input("Type your message here...")

    if user_query:

        # soup_data = soup(url_input)
        # clean_text = remove_html_script_style_tags(str(soup_data))
        # chunks = split_documents_into_chunks(clean_text)
        # index = upsert_to_pinecone(chunks) 


        st.session_state.messages.append({"role": "user", "content": user_query})

        query_embedding = emb_model.encode(user_query)
        query_embedding_list = query_embedding.tolist()
        print(user_query)

        query_response = index.query(
            vector=query_embedding_list,
            top_k=3,
            include_metadata=True,
            include_values=True
        )
        relevant_text = query_response['matches'][0]['metadata']['text']
        chat_history = []
        prompt = make_rag_prompt(user_query, relevant_text)
        answer = generate_answer(chat_history, prompt)
        
        # Generate assistant response (here simply echo for demo)
        assistant_response = f"Echo: {answer}"
        
        # Append assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": answer})


    st.button("Clear message", on_click = clear_conversation)
    display_chat_history()


