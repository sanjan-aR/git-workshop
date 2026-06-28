
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

Answer the user's question using ONLY the provided context.

Rules:
1. Read the entire context carefully before answering.
2. If the answer is present, answer directly, accurately, and professionally.
3. Do not mention "the context" or "the provided documents" in your answer.
4. Keep the answer concise while including ALL relevant policy details.
5. If the answer includes numbers, dates, durations, eligibility, or policy names, include them exactly as given.
6. If multiple relevant pieces of information exist, combine them in the same logical order as they appear in the policy.
7. Only reply with:
"I couldn't find this information in the Zyro Dynamics HR policy documents."
if the information is completely absent from the retrieved context.


Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_bot(question):

    docs = retriever.invoke(question)

    docs = retriever.invoke(question)

    print("=" * 80)
    print("QUESTION:", question)
    
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
        "question": question
    })

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

