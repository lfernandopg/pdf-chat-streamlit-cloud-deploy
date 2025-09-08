import os
import time
import locale
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

# --- Configuración de idiomas ---
LANGUAGES = {
    "es": {
        "title": "💬 Chatea con tu PDF",
        "subtitle": "Sube un PDF y haz preguntas sobre su contenido usando IA",
        "upload_section": "📄 Carga tu Documento",
        "upload_label": "Arrastra y suelta tu archivo PDF aquí o haz clic en 'Browse Files' para cargarlo",
        "chat_section": "💭 Conversación",
        "settings": "⚙️ Configuración",
        "advanced_settings": "🔧 Ajustes Avanzados",
        "model_selection": "🤖 Modelo de IA",
        "select_model": "Selecciona el modelo:",
        "process_button": "🚀 Procesar Documento",
        "processing": "🔄 Procesando tu PDF...",
        "success": "✅ PDF procesado exitosamente. ¡Ya puedes hacer preguntas!",
        "error_processing": "❌ Error al procesar el PDF:",
        "error_token": "⚠️ Debes configurar el token de HuggingFace en .streamlit/secrets.toml",
        "warning_upload": "⚠️ Por favor, carga y procesa un PDF primero. Para esto revisa el panel lateral",
        "chat_placeholder": "Escribe tu pregunta sobre el PDF...",
        "thinking": "🤔 Analizando documento...",
        "language": "🌐 Idioma",
        "stats_title": "📊 Estadísticas del Documento",
        "pages": "Páginas",
        "chunks": "Fragmentos procesados",
        "technologies": "Tecnologías utilizadas:",
        "model_info": "Modelo actual",
        "clear_chat": "🗑️ Limpiar Chat",
        "download_chat": "💾 Guardar Conversación",
        "download_btn": "Descargar",
        "chat_cleared": "✅ Chat limpio",
        "no_messages_download": "⚠️ No hay mensajes para descargar",
        "about": "ℹ️ Acerca de",
        "about_text": "Esta aplicación utiliza tecnología RAG (Retrieval Augmented Generation) para responder preguntas sobre documentos PDF usando modelos de lenguaje avanzados. Sugerencia: si no sabes que PDF subir, puedes probar con tu CV 😉",
    },
    "en": {
        "title": "💬 Chat with your PDF",
        "subtitle": "Upload a PDF and ask questions about its content using AI",
        "upload_section": "📄 Upload your Document",
        "upload_label": "Drag and drop your PDF file here",
        "chat_section": "💭 Conversation",
        "settings": "⚙️ Settings",
        "advanced_settings": "🔧 Advanced Settings",
        "model_selection": "🤖 AI Model",
        "select_model": "Select model:",
        "process_button": "🚀 Process Document",
        "processing": "🔄 Processing your PDF...",
        "success": "✅ PDF processed successfully. You can now ask questions!",
        "error_processing": "❌ Error processing PDF:",
        "error_token": "⚠️ You must configure the HuggingFace token in .streamlit/secrets.toml",
        "warning_upload": "⚠️ Please upload and process a PDF first. For this check the sidebar",
        "chat_placeholder": "Ask a question about the PDF...",
        "thinking": "🤔 Analyzing document...",
        "language": "🌐 Language",
        "stats_title": "📊 Document Statistics",
        "pages": "Pages",
        "technologies": "Technologies used:",
        "chunks": "Processed chunks",
        "model_info": "Current model",
        "clear_chat": "🗑️ Clear Chat",
        "download_chat": "💾 Save Chat",
        "download_btn": "Download",
        "chat_cleared": "✅ Chat cleared",
        "no_messages_download": "⚠️ No messages to download",
        "about": "ℹ️ About",
        "about_text": "This application uses RAG (Retrieval Augmented Generation) technology to answer questions about PDF documents using advanced language models. Tip: If you don't know which PDF to upload, you can try your CV 😉",
    }
}

def detect_system_language():
    try:
        system_locale = locale.getlocale()[0]
        print("Locale:", system_locale)
        if system_locale and system_locale.startswith('es'):
            return 'es'
        else:
            return 'en'
    except:
        return 'en'

def get_text(key, lang):
    return LANGUAGES[lang].get(key, key)

def process_prompt(prompt):
    if not st.session_state.pdf_processed or not st.session_state.rag_chain:
        st.warning(get_text("warning_upload", st.session_state.language))
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    with chat_container:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            with st.spinner(get_text("thinking", st.session_state.language)):
                try:
                    result = st.session_state.rag_chain.invoke({"query": prompt})
                    answer = result["result"]
                    words = answer.split()
                    for i, word in enumerate(words):
                        full_response += word + " "
                        if i % 3 == 0:
                            time.sleep(0.1)
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"Error: {e}"
                    placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Configuración inicial ---
load_dotenv()
st.set_page_config(page_title="PDF Chat AI", page_icon="💬", layout="wide", initial_sidebar_state="expanded")

# --- CSS personalizado ---
st.markdown("""
<style>
    .main-header { text-align:center; padding:2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color:white; border-radius:10px; margin-bottom:2rem; }
    .stats-container { background: #f8f9fa; padding:1rem; border-radius:10px; border-left:4px solid #667eea; margin:1rem 0; }
    .chat-container { border-radius:10px; padding:1rem; box-shadow:0 2px 4px rgba(0,0,0,0.1); margin:1rem 0; max-height:400px; overflow-y:auto; }
    .input-container { position:sticky; bottom:0; background:white; padding:1rem; z-index:100; border-top:1px solid #e9ecef; }
    .stChatInput { border-radius:25px !important; box-shadow:0 -2px 8px rgba(0,0,0,0.1) !important; }
    .stChatInput:focus { box-shadow: 0 0 5px rgba(0, 123, 255, 0.5); outline: none; }
    .element-container:has(.stChatInput) { margin-top:0 !important; padding:0 !important; }
    .sidebar-section { background-color: rgba(250, 250, 250, 0.2); padding:0px; border-radius:10px; margin:1rem 0; border:1px solid #808080; }
    .stButton > button { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color:white; border:none; border-radius:20px; padding:0.5rem 2rem; font-weight:bold; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow:0 4px 8px rgba(0,0,0,0.2); }
    .uploadedFile { border:2px dashed #667eea; border-radius:10px; padding:2rem; text-align:center; background:#f8f9fa; }
    .stChatMessage { border-radius:10px; padding:1rem; margin:0.5rem 0; box-shadow:0 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- Estado de la sesión ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "meta-llama/Llama-3.2-3B-Instruct:together"
if "language" not in st.session_state:
    st.session_state.language = detect_system_language()
if "document_stats" not in st.session_state:
    st.session_state.document_stats = {}

HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error(get_text("error_token", st.session_state.language))
    st.stop()

# --- Sidebar completo ---
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    if st.session_state.language == "es": 
        st.subheader(get_text("language", st.session_state.language))
    else:
        st.subheader(get_text("language", st.session_state.language))
    language_options = {"🇪🇸 Español": "es", "🇺🇸 English": "en"}
    selected_lang_display = st.selectbox(
        "",
        options=list(language_options.keys()),
        index=list(language_options.values()).index(st.session_state.language)
    )
    st.session_state.language = language_options[selected_lang_display]
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header(get_text("upload_section", st.session_state.language))
    uploaded_file = st.file_uploader(get_text("upload_label", st.session_state.language), type=["pdf"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        if st.button(get_text("process_button", st.session_state.language), use_container_width=True):
            with st.spinner(get_text("processing", st.session_state.language)):
                tmp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    loader = PyPDFLoader(tmp_file_path)
                    documents = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    texts = splitter.split_documents(documents)
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vector_store = FAISS.from_documents(texts, embeddings)
                    llm = ChatOpenAI(base_url="https://router.huggingface.co/v1",
                                     model=st.session_state.selected_model,
                                     temperature=0.7, max_tokens=512, api_key=HF_TOKEN)
                    prompt_template = """Usa el siguiente contexto para responder la pregunta de manera clara y precisa.
Si no tienes suficiente información en el contexto, indícalo claramente.

Contexto: {context}

Pregunta: {question}

Respuesta detallada:""" if st.session_state.language == "es" else """Use the following context to answer the question clearly and precisely.
If you don't have enough information in the context, indicate it clearly.

Context: {context}

Question: {question}

Detailed answer:"""
                    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                                           retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                                                           chain_type_kwargs={"prompt": prompt})
                    st.session_state.document_stats = {
                        "pages": len(documents),
                        "chunks": len(texts),
                        "filename": uploaded_file.name
                    }
                    st.session_state.rag_chain = rag_chain
                    st.session_state.pdf_processed = True
                    st.session_state.messages = []
                    st.success(get_text("success", st.session_state.language))
                except Exception as e:
                    st.error(f"{get_text('error_processing', st.session_state.language)} {e}")
                finally:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

    if not st.session_state.pdf_processed:
        with st.expander(get_text("advanced_settings", st.session_state.language)):
            st.subheader(get_text("model_selection", st.session_state.language))
            model_options = ["meta-llama/Llama-3.2-3B-Instruct:together",
                             "mistralai/Mistral-7B-Instruct-v0.3"]
            st.session_state.selected_model = st.selectbox(get_text("select_model", st.session_state.language),
                                                           model_options,
                                                           index=model_options.index(st.session_state.selected_model))

    if st.session_state.pdf_processed and st.session_state.document_stats:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader(get_text("stats_title", st.session_state.language))
        stats = st.session_state.document_stats
        st.metric(get_text("pages", st.session_state.language), stats.get("pages", 0))
        st.metric(get_text("chunks", st.session_state.language), stats.get("chunks", 0))
        st.info(f"📄 {stats.get('filename', 'N/A')}")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.pdf_processed:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        if st.button(get_text("clear_chat", st.session_state.language)):
            st.session_state.messages = []
            st.success(get_text("chat_cleared", st.session_state.language))
        if st.button(get_text("download_chat", st.session_state.language)):
            if st.session_state.messages:
                chat_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
                st.download_button(label=get_text("download_btn", st.session_state.language),
                                   data=chat_text,
                                   file_name="chat_con_pdf.txt",
                                   mime="text/plain")
            else:
                st.warning(get_text("no_messages_download", st.session_state.language))
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander(get_text("about", st.session_state.language)):
        st.info(get_text("about_text", st.session_state.language))
        st.markdown(f"**{get_text("technologies", st.session_state.language)}**")
        st.markdown("- 🤖 LangChain")
        st.markdown("- 🔍 FAISS Vector Store")
        st.markdown("- 🤗 HuggingFace Models")
        st.markdown("- ⚡ Streamlit")

# --- Header ---
st.markdown(f"""
<div class="main-header">
    <h1>{get_text("title", st.session_state.language)}</h1>
    <p style="font-size:1.2rem; margin-top:0.5rem; opacity:0.9;">
        {get_text("subtitle", st.session_state.language)}
    </p>
</div>
""", unsafe_allow_html=True)

# --- Área principal de chat ---
col1, col2 = st.columns([3, 1])
with col1:
    st.header(get_text("chat_section", st.session_state.language))
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    input_container = st.container()
    with input_container:
        if prompt := st.chat_input(get_text("chat_placeholder", st.session_state.language)):
            process_prompt(prompt)

with col2:
    if st.session_state.pdf_processed:
        st.markdown("### 🎯 " + ("Sugerencias" if st.session_state.language == "es" else "Suggestions"))
        suggestions = ["📝 Resume el documento", "🔍 ¿Cuáles son los puntos clave?",
                       "📊 Extrae datos importantes", "❓ Explica conceptos complejos"] if st.session_state.language == "es" else \
                      ["📝 Summarize the document", "🔍 What are the key points?",
                       "📊 Extract important data", "❓ Explain complex concepts"]
        for suggestion in suggestions:
            if st.button(suggestion, use_container_width=True, key=f"suggestion_{suggestion}"):
                process_prompt(suggestion[2:])
    else:
        st.info("📤 " + ("Sube un PDF para comenzar" if st.session_state.language == "es" else "Upload a PDF to start"))
