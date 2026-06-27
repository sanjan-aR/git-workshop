
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
        "k":3,
        "fetch_k":10
    }
)

# ---------------------------
# Prompt
# ---------------------------

prompt = ChatPromptTemplate.from_template("""
You are the official HR Assistant for Zyro Dynamics.

Use ONLY the provided context to answer the user's question.

Rules:
1. Never make up information.
2. Answer only from the provided context.
3. If the answer is not available in the context, reply exactly:

"Based on the available Zyro Dynamics HR policy documents, I couldn't find information that answers this question."

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

