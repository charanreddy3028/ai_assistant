import os
import re
import glob
from dotenv import load_dotenv
import streamlit as st
# Note: No DocxDocument needed anymore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LCDocument # Langchain Document
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever

# ========== LOAD ENV VARIABLES ==========
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ========== FILE PATHS CONFIGURATION ==========
# Main script/checklist file
CHECKLIST_PATH = "intr_scripts.txt" # <--- YOUR MAIN SCRIPT FILE
# Directory containing supplementary training call transcripts or examples
TRAINING_CALLS_DIR = "Training calls" # <--- FOLDER WITH .txt FILES

# ========== CONFIGURE GOOGLE (LANGCHAIN & GENERATIVEAI) ==========
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
else:
    st.error("🚨 GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
    st.stop()

# ========== HELPER FUNCTIONS ==========

def clean_text(text):
    """Cleans text by removing extra newlines and excessive whitespace."""
    text = text.replace("\n", " ") # Replace newline with space
    return re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space

def load_text_file_content(path, source_type="unknown"):
    """Loads and cleans content from a single TXT file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            st.warning(f"⚠️ The file '{path}' is empty or contains only whitespace.")
            return None
        cleaned_content = clean_text(content)
        # Create a single Langchain Document for the entire file content
        return LCDocument(page_content=cleaned_content, metadata={"source": os.path.basename(path), "type": source_type})
    except FileNotFoundError:
        st.error(f"🚨 Error: The file was not found at '{path}'.")
        return None
    except Exception as e:
        st.error(f"🚨 Error loading TXT file '{path}': {e}")
        return None

def load_training_call_files(directory_path):
    """Loads all .txt files from a specified directory."""
    docs = []
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    if not txt_files:
        st.warning(f"⚠️ No .txt files found in the directory '{directory_path}'.")
        return []
    for filepath in txt_files:
        doc = load_text_file_content(filepath, source_type="training_call")
        if doc:
            docs.append(doc)
    st.info(f"📚 Loaded {len(docs)} documents from '{directory_path}'.")
    return docs

def build_vector_store(checklist_doc, training_docs):
    """Builds a FAISS vector store from the checklist and training documents."""
    all_docs = []
    if checklist_doc:
        all_docs.append(checklist_doc)
    if training_docs:
        all_docs.extend(training_docs)

    if not all_docs:
        st.warning("⚠️ No documents loaded. Cannot build vector store.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(all_docs)

    if not split_docs:
        st.warning("⚠️ Documents could not be split into chunks. Cannot build vector store.")
        return None

    # --- Metadata assignment (Example: based on source type) ---
    # You can add more sophisticated metadata based on content if needed.
    # The example metadata from the NxtWave script (Loan Resistance, EMI Info) has been removed
    # as it was specific to that document's content.
    # The 'source' and 'type' metadata are already set during loading.
    # --- End Metadata assignment ---

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.info(f"ℹ️ Attempting to build FAISS vector store with {len(split_docs)} document chunks using Google Embeddings...")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        st.success("✅ FAISS vector store built successfully!")
        return vector_store
    except Exception as e:
        st.error(f"🚨 Error building FAISS vector store with Google Embeddings: {e}")
        st.error("   This might be due to API key issues, model compatibility, or network problems.")
        st.error("   Ensure your GOOGLE_API_KEY is correct and has Gemini API enabled.")
        st.error("   Also check if the embedding model name ('models/embedding-001') is valid and supported.")
        return None

# ========== LANGCHAIN SETUP ==========
@st.cache_resource # Caching the chain for performance
def load_chain():
    """Loads the conversational retrieval chain."""
    checklist_doc = load_text_file_content(CHECKLIST_PATH, source_type="checklist_script")
    training_call_docs = load_training_call_files(TRAINING_CALLS_DIR)

    # Proceed even if only one source is available, but warn if checklist is missing
    if not checklist_doc and not training_call_docs:
        st.error("🚨 Critical: Neither the checklist/script file nor any training call files could be loaded. The assistant cannot function.")
        return None
    if not checklist_doc:
        st.warning(f"⚠️ Checklist/script file '{CHECKLIST_PATH}' not loaded. Assistant will rely only on training call data.")

    vector_store = build_vector_store(checklist_doc, training_call_docs)
    if not vector_store:
        return None

    class TieredRetriever(VectorStoreRetriever):
        def get_relevant_documents(self, query):
            all_docs = self.vectorstore.similarity_search(query, k=10)
            checklist_docs = [doc for doc in all_docs if doc.metadata.get("type") == "checklist_script"]
            training_docs = [doc for doc in all_docs if doc.metadata.get("type") == "training_call"]

            # Priority: checklist first, then training, then fallback
            if checklist_docs:
                return checklist_docs
            elif training_docs:
                return training_docs
            else:
                return all_docs  # fallback (rare)

    retriever = TieredRetriever(vectorstore=vector_store)

    # !!! CUSTOMIZE THIS PROMPT THOROUGHLY BASED ON YOUR intr_scripts.txt CONTENT !!!
    custom_prompt_template = """
    You are an expert, friendly, and helpful assistant. Your goal is to help users understand and correctly apply the guidelines and scripts provided in the main script document (often referred to as '{checklist_filename}'), using examples from supplementary training materials (like call transcripts) for clarification where appropriate.

    **IMPORTANT: Follow these rules in order:**

    **Rule 1: Handle ONLY THESE EXACT General Conversational Turns (Without using CONTEXT)**
    Carefully examine the user's current input "{question}". Check if it EXACTLY MATCHES one of these patterns:
       - IF the input is EXCLUSIVELY a greeting (EXACTLY matching one of: "hi", "hello", "hey", "good morning", "good evening") with NO additional content:
         Respond with a warm greeting like "Hello! How can I assist you today with the scripts/guidelines?" or "Hi there! What can I help you with?". **DO NOT use the CONTEXT below for this response.**
       - IF the input is EXCLUSIVELY an expression of gratitude (EXACTLY matching one of: "thank you", "thanks", "thanks a lot", "appreciate it") with NO additional content:
         Acknowledge their gratitude warmly and offer further assistance. For example: "You're very welcome! I'm happy to help. Is there anything else you'd like to know?" or "My pleasure! Do you have any other questions?". **DO NOT use the CONTEXT below for this response.**
       - IF the input is EXCLUSIVELY a simple affirmative/understanding (EXACTLY matching one of: "okay", "got it", "alright", "fine", "ok") with NO additional content:
         Briefly acknowledge it and gently prompt if they have more questions. For example: "Okay, great! Please let me know if anything else comes to mind." or "Understood. Feel free to ask if you have more questions.". **DO NOT use the CONTEXT below for this response.**
       - IF the input is EXCLUSIVELY a farewell (EXACTLY matching one of: "bye", "goodbye", "no more questions", "that's all", "nothing else") with NO additional content:
         Respond with a polite closing. For example: "Thank you for chatting. Have a wonderful day!" or "Alright, it was a pleasure assisting you. Goodbye!". **DO NOT use the CONTEXT below for this response.**

    **Rule 2: For ALL OTHER inputs, including ANY questions, statements, or complex messages, use CONTEXT and CHAT HISTORY for Answering**
    For ANY input that does not EXACTLY match one of the simple patterns specified in Rule 1, proceed with the following instructions using the provided CONTEXT and CHAT HISTORY:

    **Overall Tone (for context-based answers):**
    Be helpful, clear, professional, and empathetic.

    **Answering Specific Questions / Handling Statements using Provided CONTEXT:**
    1.  Your ONLY source of information for these answers is the CONTEXT provided below (derived from '{checklist_filename}' and training materials). DO NOT use external knowledge or make up answers.
    2.  If the answer cannot be found within the CONTEXT, state: "I'm sorry, I don't have that specific information in my current documents. Is there something else I can help you find?"

    3.  **CRITICAL: SCRIPTED DIALOGUE HANDLING (EXAMPLE - ADAPT THIS TO YOUR `intr_scripts.txt`)**
        Carefully examine the user's current input "{question}" and the recent {chat_history}.
        **(This is an example structure for handling specific objections or scenarios from your script.
        You MUST adapt the trigger phrases, the question to ask, and the rebuttal/explanation
        to match the actual content and requirements of your '{checklist_filename}'.)**

        A.  **STEP 1: ASKING FOR CLARIFICATION (Trigger: e.g., User expresses hesitation about a process)**
            IF the user's current input "{question}" is "I'm not sure about this process" or a very similar statement of hesitation defined in your script (e.g., "I don't want to provide [specific info]")
            AND IF your IMMEDIATELY PRECEDING response in {chat_history} was NOT "Could you tell me what your main concerns are regarding this?" (i.e., you are not already in the middle of this script):
            THEN your *entire and only response for this turn* MUST be: "Could you tell me what your main concerns are regarding this?"
            DO NOT add any other information. DO NOT try to provide the rebuttal yet. Just ask this question and wait for the user's response.

        B.  **STEP 2: DELIVERING THE SCRIPTED RESPONSE (Trigger: User provides concerns after you asked)**
            IF your IMMEDIATELY PRECEDING response in {chat_history} WAS "Could you tell me what your main concerns are regarding this?"
            AND the user's current input "{question}" is their answer to that question (i.e., they are stating their concerns)
            AND the retrieved {context} contains the script for handling this specific hesitation (e.g., how to explain the benefits or security of the process):
            THEN your response MUST be to deliver the relevant explanation, drawing details from the {context}. For example:
            "I understand your concerns. [Insert scripted explanation from your intr_scripts.txt here, addressing common points like: 'Many people initially wonder about X, but this process is designed to Y...' or 'The reason we ask for Z is to ensure ABC...']. This approach helps ensure [benefit described in script]. Does this clarification help address some of your concerns?"
            (Ensure you draw directly from the relevant parts of your '{checklist_filename}' for this explanation).

    4.  **GENERAL DIRECT MATCH HANDLING (For other queries matching script headings/points):**
        IF the user's question is NOT a trigger for a special scripted dialogue (as handled in point 3A)
        AND the user's question or a very similar statement appears as a specific point, heading, or question explicitly addressed within the retrieved CONTEXT from '{checklist_filename}' or training materials:
        THEN follow these sub-steps:
            a. Acknowledge the source: "Regarding your question about '[user's statement]', the guidelines provide the following information..."
            b. Clearly explain THE GUIDANCE OR SCRIPTED RESPONSE provided in that specific section of the CONTEXT. Focus on explaining *what the script/guideline says*.

    5.  **GENERAL QUESTION HANDLING (All other questions):**
        IF the user's question is not covered by point 3 (Scripted Dialogue) or point 4 (General Direct Match):
        THEN synthesize a comprehensive answer from all relevant information within the CONTEXT. If possible, mention if the information comes primarily from the main script or a training example.

    6.  **PRESENTATION (For points 3B, 4, and 5):**
        a. Present information clearly, concisely, and in a helpful, professional tone.
        b. Paraphrase and explain information from the CONTEXT; don't just copy large blocks unless it's a direct quote from a script.
        c. Maintain your persona as a helpful assistant for the scripts/guidelines.

    CONTEXT (Only use if Rule 1 is NOT met. Derived from '{checklist_filename}' and training materials):
    {context}

    Conversation History (CRITICAL for SCRIPTED DIALOGUE HANDLING if Rule 1 is NOT met):
    {chat_history}

    User Question:
    {question}

    Answer (Follow Rule 1 first, then Rule 2 if applicable):
    """
    custom_prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "chat_history", "question"],
        partial_variables={"checklist_filename": os.path.basename(CHECKLIST_PATH)}
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", # or "gemini-pro"
            temperature=0.2, # Lower for more factual, higher for more creative
            convert_system_message_to_human=True
        )
        st.info("ℹ️ ChatGoogleGenerativeAI model (gemini-1.5-flash-latest) initialized.")
    except Exception as e:
        st.error(f"🚨 Error initializing ChatGoogleGenerativeAI: {e}")
        return None

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer' # Ensure this matches what the chain actually outputs as the answer
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
    )
    st.success("✅ ConversationalRetrievalChain loaded successfully!")
    return chain

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="AI Script Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI-Powered Script & Guideline Assistant")
st.caption(f"I'm here to help you with the '{os.path.basename(CHECKLIST_PATH)}' script and related training materials. Just ask!")
st.markdown("---")

# Display loaded file information
if os.path.exists(CHECKLIST_PATH):
    st.markdown(f"**Primary Knowledge Source:** `{os.path.basename(CHECKLIST_PATH)}`.")
else:
    st.error(f"🚨 **Critical Error:** Main script file `{CHECKLIST_PATH}` not found. The assistant may not function correctly.")

if os.path.isdir(TRAINING_CALLS_DIR) and glob.glob(os.path.join(TRAINING_CALLS_DIR, "*.txt")):
    st.markdown(f"**Supplementary Materials:** Files from `{TRAINING_CALLS_DIR}` folder.")
else:
    st.warning(f"⚠️ **Note:** Training calls directory `{TRAINING_CALLS_DIR}` not found or is empty. Assistant will rely solely on `{os.path.basename(CHECKLIST_PATH)}` if available.")

st.markdown("The vector store is built using FAISS with embeddings from Google's models.")

# !!! UPDATE THESE EXAMPLE QUESTIONS TO BE RELEVANT TO YOUR intr_scripts.txt !!!
example_questions = [
    "What is the first step in the introduction script?",
    "How should I respond if a customer objects to [common objection]?",
    "Can you give an example of a good closing statement?",
    "What are the key points to cover about [specific product/service]?",
    "Explain the process for [specific task from script]."
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_query_from_sidebar" not in st.session_state:
    st.session_state.user_query_from_sidebar = None

with st.sidebar:
    st.subheader("💡 Example Questions")
    st.write("Click a question to ask the assistant:")
    for i, eq_question in enumerate(example_questions):
        if st.button(eq_question, key=f"sidebar_eq_{i}", use_container_width=True):
            st.session_state.user_query_from_sidebar = eq_question
            st.rerun() # Rerun to process the click immediately

qa_chain = load_chain()

if not qa_chain:
    st.error("🚨 AI Chain could not be loaded. Please check file paths, API key, document content, and console logs for errors. The application cannot proceed.")
    st.stop()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query_to_process_this_run = None
if st.session_state.user_query_from_sidebar:
    query_to_process_this_run = st.session_state.user_query_from_sidebar
    st.session_state.user_query_from_sidebar = None # Reset after use

typed_query = st.chat_input("Ask your question about the script or guidelines...")
if typed_query:
    query_to_process_this_run = typed_query

if query_to_process_this_run:
    st.session_state.messages.append({"role": "user", "content": query_to_process_this_run})
    with st.chat_message("user"):
        st.markdown(query_to_process_this_run)

    try:
        with st.spinner("🤖 Thinking with Gemini... Please wait."):
            # Make sure the input to the chain matches what it expects
            # The ConversationalRetrievalChain expects a dictionary with a 'question' key
            response = qa_chain.invoke({"question": query_to_process_this_run})

        answer = response.get("answer", "Sorry, I couldn't find an answer.")
        source_documents = response.get("source_documents", [])

        with st.chat_message("assistant"):
            st.markdown(answer)

            if source_documents and len(str(answer)) > 30 and "don't have that specific information" not in str(answer).lower() : # Show sources if answer is substantial
                with st.expander("See sources retrieved (from FAISS with Google Embeddings)"):
                    for i, doc in enumerate(source_documents):
                        doc_source = doc.metadata.get('source', 'Unknown source')
                        doc_type = doc.metadata.get('type', 'unknown_type')
                        st.markdown(f"**Source {i + 1} (File: {doc_source}, Type: {doc_type}):**")
                        st.caption(f"```{doc.page_content[:300]}...```") # Display first 300 chars

        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        error_message = f"🚨 Error during QA chain execution with Gemini: {e}"
        st.error(error_message)
        st.exception(e) # Provides full traceback in the terminal and optionally in UI
        with st.chat_message("assistant"):
            st.markdown(f"Sorry, I encountered an error processing your request. Please check the console for details.")
        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error. Details: {e}"})