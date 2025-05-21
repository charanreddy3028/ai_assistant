# Updated main.py with Gemini-native multilingual TTS via Google SDK (fixing response mime type)

import os
import re
import glob
import io
import json
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LCDocument
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from deep_translator import GoogleTranslator
import google.generativeai as genai

# ========== LOAD ENV VARIABLES ==========
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ========== FILE PATHS CONFIGURATION ==========
CHECKLIST_PATH = "intr_scripts.txt"
TRAINING_CALLS_DIR = "Training calls"
CONVERSATIONAL_MAP_PATH = "conversational_map.json"

# ========== API KEY CHECK ==========
if not GOOGLE_API_KEY:
    st.error("🚨 GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# ========== HELPERS ==========
def clean_text(text):
    return re.sub(r'\s+', ' ', text.replace("\n", " ")).strip()

def load_text_file_content(path, source_type="unknown"):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            st.warning(f"⚠️ The file '{path}' is empty.")
            return None
        return LCDocument(page_content=clean_text(content), metadata={"source": os.path.basename(path), "type": source_type})
    except Exception as e:
        st.error(f"🚨 Error loading {path}: {e}")
        return None

def load_training_call_files(directory_path):
    docs = []
    for filepath in glob.glob(os.path.join(directory_path, "*.txt")):
        doc = load_text_file_content(filepath, source_type="training_call")
        if doc:
            docs.append(doc)
    return docs

def build_vector_store(checklist_doc, training_docs):
    all_docs = [doc for doc in [checklist_doc] + training_docs if doc]
    if not all_docs:
        st.warning("⚠️ No documents found to build vector store.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(all_docs)

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.from_documents(split_docs, embeddings)
    except Exception as e:
        st.error(f"🚨 FAISS build failed: {e}")
        return None

# ========== TIERED RETRIEVER ==========
class TieredRetriever(VectorStoreRetriever):
    def _get_relevant_documents(self, query):
        docs = self.vectorstore.similarity_search(query, k=10)
        checklist = [d for d in docs if d.metadata.get("type") == "checklist_script"]
        training = [d for d in docs if d.metadata.get("type") == "training_call"]
        return checklist or training or docs

# ========== Conversational Post-Processor ==========
def load_conversational_map():
    try:
        with open(CONVERSATIONAL_MAP_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def make_conversational(text, language):
    conv_map = load_conversational_map().get(language.lower(), {})
    for formal, casual in conv_map.items():
        text = re.sub(rf"\\b{re.escape(formal)}\\b", casual, text)
    return text

# ========== Gemini-native TTS ==========
def synthesize_audio_with_gemini(text, language):
    st.info("🔁 Generating audio with Gemini-native model...")

    if language != "English":
        lang_code = {"Hindi": "hi", "Telugu": "te"}.get(language, "en")
        try:
            text = GoogleTranslator(source='auto', target=lang_code).translate(text)
            text = make_conversational(text, language)
            st.info(f"🌐 Conversational {language}: {text}")
        except Exception as e:
            st.error(f"Translation failed: {e}")
            return None

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-native-audio-dialog")
        response = model.generate_content(text, generation_config={"response_mime_type": "application/json"})

        # Handle response which might contain an audio URL or fallback
        if hasattr(response, 'audio'):
            return io.BytesIO(response.audio)
        elif hasattr(response, 'parts') and isinstance(response.parts[0], bytes):
            return io.BytesIO(response.parts[0])
        elif hasattr(response, 'text') and "http" in response.text:
            import requests
            res = requests.get(response.text)
            return io.BytesIO(res.content) if res.status_code == 200 else None
        else:
            st.error("Gemini response did not include audio.")
            return None
    except Exception as e:
        st.error(f"❌ Gemini TTS Error: {e}")
        return None

# ========== LOAD CHAIN ==========
@st.cache_resource
def load_chain():
    checklist_doc = load_text_file_content(CHECKLIST_PATH, "checklist_script")
    training_docs = load_training_call_files(TRAINING_CALLS_DIR)
    vector_store = build_vector_store(checklist_doc, training_docs)
    if not vector_store:
        return None

    retriever = TieredRetriever(vectorstore=vector_store)

    prompt_template = """
    You are a high-performing sales agent who consistently converts 5–10 leads per day. You are confident, empathetic, and persuasive, but always respectful.

    If the input is a greeting, gratitude, or farewell, respond naturally without using context.

    If the input is a question or objection:
    - Use the CONTEXT and CHAT HISTORY if available.
    - If no relevant CONTEXT is found, DO NOT say "I don't know".
    - Instead, use your best judgment as a top sales agent to respond confidently and persuasively.
    - Handle objections like "I'm not interested" or "I'm not sure" like you're on a live call.
    - DO NOT answer sensitive or company-critical questions. Redirect politely.

    CONTEXT:
    {context}

    CHAT HISTORY:
    {chat_history}

    USER:
    {question}

    YOUR RESPONSE:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.2
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

# ========== UI ==========
st.set_page_config(page_title="AI Script Assistant", page_icon="🤖")
st.title("🤖 AI-Powered Script & Guideline Assistant")

language = st.selectbox("🗣️ Select playback language:", ["English", "Hindi", "Telugu"], index=0)

qa_chain = load_chain()
if not qa_chain:
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "speech_attempts" not in st.session_state:
    st.session_state.speech_attempts = {}
if "play_flags" not in st.session_state:
    st.session_state.play_flags = {}

query = st.chat_input("Ask your question about the script or handle objections like a pro...")
if query:
    with st.spinner("🤖 Thinking like a pro sales agent..."):
        response = qa_chain.invoke({"question": query})
        answer = response.get("answer", "").strip()

        if not answer or "don't have that specific information" in answer.lower():
            fallback_prompt = f"""
            The user said: "{query}"
            No useful info was found in context.
            Respond like a high-converting sales agent: persuasive, empathetic, and confident.
            Handle the objection dynamically, like on a live sales call.
            """
            fallback = qa_chain.llm.invoke(fallback_prompt)
            answer = getattr(fallback, "content", str(fallback))

        msg_key = f"msg_{len(st.session_state.messages)}"
        st.session_state.speech_attempts[msg_key] = 0
        st.session_state.play_flags[msg_key] = False

        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": answer})

for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    content = msg["content"]
    msg_key = f"msg_{i}"

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant":
            if msg_key not in st.session_state.speech_attempts:
                st.session_state.speech_attempts[msg_key] = 0
            if msg_key not in st.session_state.play_flags:
                st.session_state.play_flags[msg_key] = False

            col1, _ = st.columns([0.1, 0.9])
            with col1:
                if st.session_state.speech_attempts[msg_key] < 2:
                    if st.button("🔊", key=f"btn_{msg_key}", help="Read out loud"):
                        st.session_state.play_flags[msg_key] = True

            if st.session_state.play_flags[msg_key] and st.session_state.speech_attempts[msg_key] < 2:
                st.info("🔔 Initiating TTS...")
                audio = synthesize_audio_with_gemini(content, language)
                if audio:
                    st.audio(audio, format="audio/mp3", start_time=0)
                    st.session_state.speech_attempts[msg_key] += 1
                else:
                    st.error("🔇 Could not generate audio.")
                st.session_state.play_flags[msg_key] = False
            elif st.session_state.speech_attempts[msg_key] >= 2:
                st.caption("🔇 Playback limit reached for this message.")
