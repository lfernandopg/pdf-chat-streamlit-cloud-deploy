import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# --- Configuración inicial ---
load_dotenv()
st.set_page_config(page_title="Chatea con tu PDF 💬", page_icon="💬", layout="centered")

st.title("Chatea con tu PDF 💬")

# --- Estado de la sesión ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "meta-llama/Llama-3.2-3B-Instruct:together"

# --- HuggingFace Token ---
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Debes configurar el token de HuggingFace en .streamlit/secrets.toml")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("1. Carga tu Documento")
    uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])

    with st.expander("⚙️ Ajustes Avanzados"):
        st.subheader("Modelo de IA")
        model_options = [
            "meta-llama/Llama-3.2-3B-Instruct:together",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]
        st.session_state.selected_model = st.selectbox(
            "Selecciona el modelo:",
            model_options,
            index=model_options.index(st.session_state.selected_model),
        )

    if uploaded_file and st.button("Procesar Documento"):
        with st.spinner("Procesando tu PDF..."):
            
            tmp_file_path = None
            try:
                # 1. Cargar PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()

                # 2. Dividir en chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = splitter.split_documents(documents)

                # 3. Crear embeddings y vector store
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(texts, embeddings)

                # 4. Configurar LLM desde HuggingFaceHub (vía ChatOpenAI wrapper)
                llm = ChatOpenAI(
                    base_url="https://router.huggingface.co/v1",
                    model=st.session_state.selected_model,
                    temperature=0.7,
                    max_tokens=512,
                    api_key=HF_TOKEN,
                )

                # 5. Prompt y cadena RAG
                prompt_template = """Usa el siguiente contexto para responder la pregunta.
Si no sabes la respuesta, dilo claramente.

Contexto: {context}

Pregunta: {question}

Respuesta:"""
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                    chain_type_kwargs={"prompt": prompt},
                )

                # Guardar en sesión
                st.session_state.rag_chain = rag_chain
                st.session_state.pdf_processed = True
                st.session_state.messages = []

                st.success("✅ PDF procesado exitosamente")

            except Exception as e:
                st.error(f"Error al procesar el PDF: {e}")
            finally:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

# --- Chat ---
st.header("2. Haz tus Preguntas")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre el PDF..."):
    if not st.session_state.pdf_processed or not st.session_state.rag_chain:
        st.warning("⚠️ Por favor, carga y procesa un PDF primero.")
    else:
        # Guardar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Respuesta del modelo
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            with st.spinner("Pensando..."):
                try:
                    result = st.session_state.rag_chain.invoke({"query": prompt})
                    answer = result["result"]

                    # Efecto de escritura
                    for chunk in answer.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"Error: {e}"
                    placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
