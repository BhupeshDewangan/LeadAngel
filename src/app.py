import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from main import get_data, remove_html_script_style_tags, soup
import google.generativeai as genai
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()


from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                            # token="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")


emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                                        token=st.secrets["HF_TOKEN"])

# Access the secret using userdata.get()
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gen_model = genai.GenerativeModel(model_name="gemini-1.5-flash")


# Your text splitter function
def split_documents_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)


def upsert_to_pinecone(chunks):
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    index_name = "semantic-search-openai"

    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=384,
            metric="dotproduct",
            spec=spec
        )

    # Connect to the index
    index = pc.Index(index_name)
    time.sleep(1)

    # Prepare data for upsert
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):

        print(f"Chunk {i+1}: {chunk}\n")
        embedding_vector = emb_model.encode(chunk)  # This returns a list of 384 floats
        vectors_to_upsert.append({
            "id": f"chunk-{i+1}",
            "values": embedding_vector,
            "metadata": {
                "text": chunk
            }
        })

    # Upsert vectors into Pinecone index (in batches if large)
    index.upsert(vectors=vectors_to_upsert)

    print(f"Upserted {len(vectors_to_upsert)} chunks into Pinecone index '{index_name}'")

    return index


def generate_answer(chat_history, prompt):
    system_message = (
    "If a query lacks a direct answer, generate a response based on related features. "
    "You are a helpful and respectful sales chat assistant who answers queries relevant only to the context/provied text. "
    "Please answer all questions politely. Use a conversational tone, like you're chatting with someone, "
    "not like you're writing an email. If the user asks about anything outside of the sales data like if they ask "
    "something irrelevant, simply say, 'I can only provide answers related to the relevent text, sir."
    )

    #  Append the new prompt to the chat history
    chat_history.append(f"User: {prompt}")

    # Combine the system message with the chat history
    full_prompt = f"{system_message}\n\n" + "\n".join(chat_history) + "\nAssistant:"

    # Generate the response and add it to the chat history
    response = gen_model.generate_content(full_prompt)
    chat_history.append(f"Assistant: {response.text}")
    
    return response.text


def make_rag_prompt(query, context):
    return f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"

# if __name__ == "__main__":

#     print("model loading...")
#     emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
#                             token=os.getenv("HuggingFace_API_KEY"))

#     start_time = time.perf_counter()  # Start timing

#     soup_data = soup("https://docs.leadangel.com/en/sales-team/weighted-sales-team")
#     clean_text = remove_html_script_style_tags(str(soup_data))

#     chunks = split_documents_into_chunks(clean_text)

#     index = upsert_to_pinecone(chunks=chunks)

#     print("start")
#     a = str(input("Enter your query: "))

#     query_embedding = emb_model.encode(a)
#     print(f"Query embedding: {len(query_embedding)}")
#     query_embedding_list = query_embedding.tolist()  # convert ndarray to list


#     query_response = index.query(
#         vector=query_embedding_list,
#         top_k=3,
#         include_metadata=True, 
#         include_values=True
#     )

#     # print(f"Query response: {query_response}")

#     relevant_text = query_response['matches'][0]['metadata']['text']


#     chat_history = []

#     # Generate and print the final answer, maintaining chat history
#     prompt = make_rag_prompt(a, relevant_text)
#     answer = generate_answer(chat_history, prompt)
#     print("Answer:", answer)


#     end_time = time.perf_counter()  # End timing



    




