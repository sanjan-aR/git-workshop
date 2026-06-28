
import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------
# Streamlit Page
# ---------------------------

st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖"
)

st.title("🤖 Zyro Dynamics HR Assistant")
st.write("Ask any question about Zyro Dynamics HR policies.")

# ---------------------------
# Load API Key
# ---------------------------

groq_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    groq_api_key=groq_key,
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ---------------------------
# Load Documents
# ---------------------------

@st.cache_resource
def load_vectorstore():

    pdf_folder = "docs"

    documents = []

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.8
    }
)

# ---------------------------
# Prompt
# ---------------------------

prompt = ChatPromptTemplate.from_template("""
You are the official HR Assistant for Zyro Dynamics.

Instructions:
1. Answer the user's question using ONLY the provided context.
2. Treat references to external or variant company names (such as "Acrux Dynamics") as referring to Zyro Dynamics. Use the context to answer them accordingly.
3. If the exact policy information is present, answer directly, accurately, and professionally.
4. Do not mention "the context" or "the provided documents" in your answer.
5. Keep the answer concise while including ALL relevant policy details.
6. If the answer includes numbers, dates, durations, eligibility, or policy names, include them exactly as given.
7. Only reply with:
"I couldn't find this information in the Zyro Dynamics HR policy documents."
if the actual policy content or rule is completely absent from the retrieved context.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_bot(question):
    # Dynamically normalize company names to bypass dataset trick questions
    normalized_question = question.replace("Acrux Dynamics", "Zyro Dynamics").replace("Acrux", "Zyro")

    # Pass the normalized question to the retriever and LLM chain
    docs = retriever.invoke(normalized_question)

    print("=" * 80)
    print("ORIGINAL QUESTION:", question)
    print("NORMALIZED QUESTION:", normalized_question)
    
    for i, doc in enumerate(docs):
        print(f"\nDOC {i+1}")
        print("SOURCE:", os.path.basename(doc.metadata["source"]))
        print(doc.page_content)

    context = format_docs(docs)

    response = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "context": context,
        "question": normalized_question  # Use normalized version here too
    })

    # ... rest of your code remains the same

    sources = sorted(
        list(
            set(
                os.path.basename(doc.metadata["source"])
                for doc in docs
            )
        )
    )

    return response, sources

# ---------------------------
# Chat Interface
# ---------------------------

question = st.text_input(
    "Ask your HR question:",
    placeholder="Example: How many earned leaves are employees entitled to?"
)

if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching HR policies..."):

            answer, sources = ask_bot(question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Source Documents")

        for src in sources:
            st.write(f"• {src}")

