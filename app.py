import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# 1. Page Configuration (Sabse upar hona chahiye)
st.set_page_config(page_title="Kisan Saathi AI", page_icon="🌾", layout="wide")

# Aapki API Key (Directly integrated)
GOOGLE_API_KEY = "AIzaSyDhlG9o12Vmi8WqnOqJHrwi-3iSOhg08Sg"

# 2. Enhanced Page Styling (Attractive UI) 
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f0f4f0, #ffffff);
    }
    [data-testid="stSidebar"] {
        background-color: #1e4620;
        color: white;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #2e7d32;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: #ffd700;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar with Features
with st.sidebar:
    st.title("🌾 Kisan Saathi")
    st.info("Your AI Agriculture Expert")
    
    st.markdown("---")
    st.subheader("📁 Knowledge Management")
    pdf_docs = st.file_uploader("Upload Advisory PDFs", accept_multiple_files=True) [cite: 8]
    
    if st.button("🔄 Train AI on Documents"):
        if pdf_docs:
            with st.spinner("Processing agricultural data..."):
                # Document loading [cite: 7]
                text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                
                # Meaningful chunking [cite: 9]
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(text)
                
                # Embeddings generation & storage in FAISS [cite: 10]
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                vector_store.save_local("faiss_index")
                st.success("Analysis Complete!")
        else:
            st.error("Please upload PDFs first")

    st.markdown("---")
    language = st.selectbox("Select Language / भाषा चुनें", ["English", "Hindi (हिंदी)"]) [cite: 19]

# 4. Main Chat Interface & Dashboard
st.title("🚜 Smart Crop Advisory System")
st.caption("Accurate, document-based guidance for Indian farmers.") [cite: 19]

# Dashboard Metrics 
col1, col2, col3, col4 = st.columns(4)
col1.metric("Location", "Bhopal, MP")
col2.metric("Temp", "34°C", "Sunny")
col3.metric("Soil Type", "Black Soil")
col4.metric("Status", "Ready")

st.markdown("---")

# Prompt Engineering for Source Grounding 
def get_agri_chain():
    prompt_template = """
    You are a professional Kisan Saathi (Farmer Friend). 
    Use the context provided to answer the farmer's question precisely.
    
    Rules:
    1. Only use the provided context. If not present, say 'Mujhe maaf kijiye, par iska jawab mere pas nahi hai.' 
    2. Provide step-by-step instructions for fertilizers or pests. [cite: 19]
    3. Language: {lang_choice}
    4. Always end with a safety disclaimer.

    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY) [cite: 12]
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "lang_choice"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# 5. Handling Queries
query = st.chat_input("Ask about Pesticides, Soil, or Crop help...")

if query:
    if not os.path.exists("faiss_index"):
        st.warning("Please upload and process documents first!")
    else:
        st.chat_message("user").write(query)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Retrieval of relevant chunks [cite: 11]
        docs = db.similarity_search(query)
        chain = get_agri_chain()
        
        with st.chat_message("assistant"):
            # Answer generation using Google Gemini API [cite: 12]
            response = chain({"input_documents": docs, "question": query, "lang_choice": language}, return_only_outputs=True)
            st.write(response["output_text"])
            
            # Source-aware answers with citations [cite: 14]
            with st.expander("📚 View Evidence (Source Chunks)"):
                for i, doc in enumerate(docs):
                    st.write(f"**Source {i+1}:** {doc.page_content[:300]}...")

st.markdown("---")
st.caption("⚠️ Disclaimer: Always consult a local agri-officer for chemical usage.")
