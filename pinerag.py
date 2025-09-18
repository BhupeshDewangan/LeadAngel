import streamlit as st
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import os
from sentence_transformers import SentenceTransformer
from main import get_data, remove_html_script_style_tags, soup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure generative AI model
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gen_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Text splitter function
def split_documents_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# Upsert chunks to Pinecone
def upsert_to_pinecone(chunks, emb_model):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    index_name = "semantic-search-openai"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=384,
            metric="dotproduct",
            spec=spec
        )
    index = pc.Index(index_name)
    time.sleep(1)

    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        embedding_vector = emb_model.encode(chunk)
        vectors_to_upsert.append({
            "id": f"chunk-{i+1}",
            "values": embedding_vector,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors_to_upsert)

    return index

# Generate answer from chat history
def generate_answer(chat_history, prompt):
    system_message = (
        "If a query lacks a direct answer, generate a response based on related features. "
        "You are a helpful and respectful sales chat assistant who answers queries relevant only to the context/provied text. "
        "Please answer all questions politely. Use a conversational tone, like you're chatting with someone, "
        "not like you're writing an email. If the user asks about anything outside of the sales data like if they ask "
        "something irrelevant, simply say, 'I can only provide answers related to the relevent text, sir."
    )
    chat_history.append(f"User: {prompt}")
    full_prompt = f"{system_message}\n\n" + "\n".join(chat_history) + "\nAssistant:"
    response = gen_model.generate_content(full_prompt).text
    chat_history.append(f"Assistant: {response}")
    return response

# Format RAG prompt
def make_rag_prompt(query, context):
    return f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"

  

if __name__ == "__main__":
    emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                                        token=os.getenv("HuggingFace_API_KEY"))
    st.set_page_config(page_title="Generate Q&A from Website", page_icon="🤖")
    st.title("Generate Q&A from Website")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "AI", "content": "Hello, I am a bot. How can I help you?"},
        ]
    if "show_question_box" not in st.session_state:
        st.session_state.show_question_box = False
    
    # Sidebar for URL input and Next button
    with st.sidebar:
        st.header("Settings")
        website_url = st.text_input("Website URL", key="website_url_input")
        if st.button('Next'):
            
            # Clear chat history when URL is updated
            st.session_state.chat_history = []
            st.session_state.show_question_box = True
    
    # Check if URL is provided
    if not website_url:
        st.info("Please enter a website URL")
    else:
        # When next is clicked and URL is pasted, show question input box
        if st.session_state.show_question_box:
            user_query = st.text_input("Type your message here...", key="user_input")
            
            if user_query:
                print(user_query)
                soup_data = soup(website_url)
                clean_text = remove_html_script_style_tags(str(soup_data))
                chunks = split_documents_into_chunks(clean_text)
                index = upsert_to_pinecone(chunks, emb_model) 

                # print(soup_data)
                # print(clean_text)

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
                # st.write("Answer:", answer)

                st.session_state.chat_history.append({"role": "Human", "content": user_query})
                st.session_state.chat_history.append({"role": "AI", "content": answer})

        else:
            st.info("Click 'Next' in the sidebar to start asking questions.")

        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "AI":
                with st.chat_message("AI"):
                    st.write(message["content"])
            elif message["role"] == "Human":
                with st.chat_message("Human"):
                    st.write(message["content"])




































# if __name__ == "__main__":
#     emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
#                                         token=os.getenv("HuggingFace_API_KEY"))
#     st.sidebar.title("Input URL")
#     url_input = st.sidebar.text_input("Enter the URL")

#     user_query = st.text_input("Enter your query:")
#     print(user_query)

#     if "index" not in st.session_state:
#         st.session_state.index = False

#     if st.sidebar.button("Next"):
#         # Load model once user proceeds
        
#         soup_data = soup(url_input)
#         clean_text = remove_html_script_style_tags(str(soup_data))
#         chunks = split_documents_into_chunks(clean_text)
#         index = upsert_to_pinecone(chunks, emb_model) 

#         st.session_state.index = True
#         # emb_model = emb_model.to_empty(device)
#         # Display loading info

#         st.success("Content processed and indexed!")

#         # Show input box for query on main page
        
#     if user_query and emb_model:
#         query_embedding = emb_model.encode(user_query)
#         query_embedding_list = query_embedding.tolist()
#         print(user_query)

#         query_response = index.query(
#             vector=query_embedding_list,
#             top_k=3,
#             include_metadata=True,
#             include_values=True
#         )
#         relevant_text = query_response['matches'][0]['metadata']['text']
#         chat_history = []
#         prompt = make_rag_prompt(user_query, relevant_text)
#         answer = generate_answer(chat_history, prompt)
#         st.write("Answer:", answer)

