# LeadAngel AI Assistant – Prompt-based Documentation Search Tool

## Overview

The LeadAngel AI Assistant is a web-based, AI-powered tool that enables intuitive search and conversational interaction with LeadAngel’s internal sales team documentation. Instead of manually browsing through long pages, users can simply type a query into a Google-like prompt and receive instant, accurate answers generated from the Weighted Sales Team documentation.

This project leverages Retrieval-Augmented Generation (RAG) to combine semantic document retrieval with powerful LLM-based reasoning, ensuring responses are contextually correct, clear, and conversational.

## Features

- **Google-like Prompt Search:** Natural language queries on documentation.
- **Chat-based Interface:** Conversational history and follow-ups for refined queries.
- **Dynamic Document Scraping:** Uses BeautifulSoup to fetch documentation directly.
- **Vector Search with Pinecone:** Efficient similarity-based retrieval of documentation chunks.
- **RAG Pipeline with LangChain:** Splits, embeds, and retrieves text intelligently.
- **Generative AI Answers:** Powered by Google Generative AI (Gemini API).
- **Streamlit UI:** Lightweight, interactive web application with session state and history.
- **Secrets Management:** Secure environment handling with .env or Streamlit secrets.

## Tech Stack

- Python 3.10+
- Streamlit – UI and chat interface
- Sentence Transformers – Embeddings
- Pinecone – Vector database
- LangChain – Document splitting and RAG
- Google Generative AI (Gemini API) – Large Language Model
- BeautifulSoup / Requests – Web scraping
- Dotenv – Environment variable management

## High-Level Workflow

- **Data Acquisition:**  
  Scrapes and cleans LeadAngel documentation using BeautifulSoup & Requests.

- **Chunking & Embedding:**  
  Uses LangChain’s RecursiveCharacterTextSplitter and Sentence Transformers to embed text.

- **Vector Database Upsert:**  
  Stores all embeddings inside Pinecone for fast semantic lookup.

- **Search & Retrieval (RAG):**  
  User queries are embedded and matched against documentation chunks.

- **AI Answer Generation:**  
  Contextual knowledge + user query → passed to Gemini LLM for response synthesis.

- **Interactive UI:**  
  Streamlit interface provides chat history, reset options, and feedback.

## Setup Instructions

### Prerequisites

- Python 3.10+
- Pinecone account & API key
- Google Generative AI (Gemini) API key
- Streamlit installed

