import os
import re
from dotenv import load_dotenv
import streamlit as st
from docx import Document as DocxDocument # To avoid confusion with langchain.docstore.document.Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LCDocument # Langchain Document
from langchain.prompts import PromptTemplate


# ========== LOAD ENV VARIABLES ==========
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# !!! IMPORTANT: Set the correct path to your DOCX file here !!!
# For example: DOCX_PATH = "your_document_name.docx"
# Or an absolute path: DOCX_PATH = "/path/to/your/English_Check.docx"
DOCX_PATH = "English_Check 2.docx"# <--- MAKE SURE THIS PATH IS CORRECT

# ========== CONFIGURE GOOGLE (LANGCHAIN & GENERATIVEAI) ==========
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
else:
    st.error("🚨 GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
    st.stop()

# ========== HELPER FUNCTIONS ==========

def clean_text(text):
    """Cleans text by removing extra newlines, bullets, and excessive whitespace."""
    text = text.replace("\n", " ").replace("\u2022", "-") # Replace newline with space, bullet with hyphen
    return re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space

def load_docx_paragraphs(path):
    """Loads and cleans paragraphs from a DOCX file."""
    try:
        doc = DocxDocument(path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return [clean_text(p) for p in paragraphs if p] # Clean and filter out empty paragraphs
    except FileNotFoundError:
        st.error(f"🚨 Error: The document file was not found at '{path}'. Please ensure the DOCX_PATH is correct.")
        return []
    except Exception as e:
        st.error(f"🚨 Error loading DOCX file: {e}")
        return []

def build_vector_store(paragraphs):
    """Builds a FAISS vector store from document paragraphs."""
    if not paragraphs:
        st.warning("⚠️ No paragraphs loaded from the document. Cannot build vector store.")
        return None

    docs = [LCDocument(page_content=p) for p in paragraphs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)

    if not split_docs:
        st.warning("⚠️ Document could not be split into chunks. Cannot build vector store.")
        return None

    # --- Metadata assignment (helps in categorizing chunks) ---
    for doc_chunk in split_docs:
        content = doc_chunk.page_content.lower()
        if "don’t want loan" in content or "concerns you about a loan" in content or "don't want loan" in content:
            doc_chunk.metadata = {"section": "Loan Resistance"}
        elif "emi" in content:
            doc_chunk.metadata = {"section": "EMI Info"}
        elif "nbfc" in content:
            doc_chunk.metadata = {"section": "NBFC Rules"}
        else:
            doc_chunk.metadata = {"section": "General"}
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
    paragraphs = load_docx_paragraphs(DOCX_PATH)
    if not paragraphs:
        return None

    vector_store = build_vector_store(paragraphs)
    if not vector_store:
        return None

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    custom_prompt_template = """
    You are an expert, friendly, and emotionally appreciative NBFC agent for NxtWave. Your goal is to help users understand student loans and the NxtWave onboarding process.

    **IMPORTANT: Follow these rules in order:**

    **Rule 1: Handle ONLY THESE EXACT General Conversational Turns (Without using CONTEXT)**
    Carefully examine the user's current input "{question}". Check if it EXACTLY MATCHES one of these patterns:
       - IF the input is EXCLUSIVELY a greeting (EXACTLY matching one of: "hi", "hello", "hey", "good morning", "good evening") with NO additional content:
         Respond with a warm greeting like "Hello! How can I assist you today with NxtWave?" or "Hi there! What can I help you with regarding our student loans or onboarding process?". **DO NOT use the CONTEXT below for this response.**
       - IF the input is EXCLUSIVELY an expression of gratitude (EXACTLY matching one of: "thank you", "thanks", "thanks a lot", "appreciate it") with NO additional content:
         Acknowledge their gratitude warmly and offer further assistance. For example: "You're very welcome! I'm happy to help. Is there anything else you'd like to know?" or "My pleasure! Do you have any other questions about NxtWave?". **DO NOT use the CONTEXT below for this response.**
       - IF the input is EXCLUSIVELY a simple affirmative/understanding (EXACTLY matching one of: "okay", "got it", "alright", "fine", "ok") with NO additional content:
         Briefly acknowledge it and gently prompt if they have more questions. For example: "Okay, great! Please let me know if anything else comes to mind." or "Understood. Feel free to ask if you have more questions.". **DO NOT use the CONTEXT below for this response.**
       - IF the input is EXCLUSIVELY a farewell (EXACTLY matching one of: "bye", "goodbye", "no more questions", "that's all", "nothing else") with NO additional content:
         Respond with a polite closing. For example: "Thank you for chatting with NxtWave. Have a wonderful day!" or "Alright, it was a pleasure assisting you. Goodbye!". **DO NOT use the CONTEXT below for this response.**

    **Rule 2: For ALL OTHER inputs, including ANY questions, statements, or complex messages, use CONTEXT and CHAT HISTORY for Answering**
    For ANY input that does not EXACTLY match one of the simple patterns specified in Rule 1, proceed with the following instructions using the provided CONTEXT and CHAT HISTORY:

    **Overall Tone (for context-based answers):**
    Be helpful, clear, professional, and empathetic. Use an appreciative tone.

    **Answering Specific Questions / Handling Statements using Provided CONTEXT:**
    1.  Your ONLY source of information for these answers is the CONTEXT provided below. DO NOT use external knowledge or make up answers.
    2.  If the answer cannot be found within the CONTEXT, state: "I'm sorry, I don't have that specific information in my current documents. Is there something else I can help you find?"

    3.  **CRITICAL: SCRIPTED DIALOGUE FOR "I DON'T WANT A LOAN"**
        Carefully examine the user's current input "{question}" and the recent {chat_history}.

        A.  **STEP 1: ASKING FOR CONCERNS (Trigger: "I don't want a loan")**
            IF the user's current input "{question}" is "I don't want a loan" or a very similar statement (e.g., "I don't want to take a loan," "no loan for me")
            AND IF your IMMEDIATELY PRECEDING response in {chat_history} was NOT "What concerns you have about a loan? May I know some what clearly?" (i.e., you are not already in the middle of this script):
            THEN your *entire and only response for this turn* MUST be: "What concerns you have about a loan? May I know some what clearly?"
            DO NOT add any other information. DO NOT try to provide the rebuttal yet. Just ask this question and wait for the user's response.

        B.  **STEP 2: DELIVERING THE REBUTTAL (Trigger: User provides concerns after you asked)**
            IF your IMMEDIATELY PRECEDING response in {chat_history} WAS "What concerns you have about a loan? May I know some what clearly?"
            AND the user's current input "{question}" is their answer to that question (i.e., they are stating their concerns)
            AND the retrieved {context} contains the script for handling "Don\u2019t Want Loan" (which includes phrases like "Many parents associate loans with risk," "no-cost EMI," "Flipkart or Amazon offer for laptops," "You\u2019re not paying extra"):
            THEN your response MUST be to deliver the following explanation, drawing details from the {context} where appropriate:
            "I understand. Many parents associate loans with risk, and it's wise to be cautious. However, this is not a typical loan with interest or heavy penalties that you might be thinking of.
            Instead, this is a no-cost EMI (Equated Monthly Installment) option, processed through secure digital platforms. It's very similar to when you buy a product, say a laptop or a fridge, on a platform like Flipkart or Amazon, and they offer a 'no-cost EMI' payment plan.
            The key thing is, you\u2019re not paying any extra interest or fees. You're simply spreading the total course fee over several months to make it more manageable. Many of our parents, including farmers, homemakers, and shopkeepers, have found this approach very helpful for securing their child's future without a large upfront payment.
            Does this clarification help address some of your concerns about it being a 'loan'?"
            (Ensure you emphasize the Flipkart/Amazon analogy for no-cost EMI).

    4.  **GENERAL DIRECT MATCH HANDLING (For other queries matching document headings but not the above script):**
        IF the user's question is NOT the "I don't want a loan" trigger (as handled in point 3A)
        AND the user's question or a very similar statement appears as a specific point, heading, or question explicitly addressed within the retrieved CONTEXT (e.g., "What if NxtWave Shuts Down?"):
        THEN follow these sub-steps:
            a. Acknowledge that the NxtWave guide offers specific advice: "Regarding your question about '[user's statement]', our NxtWave material provides the following information..."
            b. Clearly explain THE GUIDANCE OR SCRIPTED RESPONSE provided in that specific section of the CONTEXT for that matched point. Focus on explaining *what the guide says*.

    5.  **GENERAL QUESTION HANDLING (All other questions):**
        IF the user's question is not covered by point 3 (Scripted Dialogue for "I don't want a loan") or point 4 (General Direct Match):
        THEN synthesize a comprehensive answer from all relevant information within the CONTEXT.

    6.  **PRESENTATION (For points 3B, 4, and 5):**
        a. Present information clearly, concisely, and in a helpful, professional tone.
        b. Paraphrase and explain information from the CONTEXT; don't just copy large blocks.
        c. Maintain your persona as an NBFC agent for NxtWave.

    CONTEXT (Only use if Rule 1 is NOT met):
    {context}

    Conversation History (CRITICAL for SCRIPTED DIALOGUE HANDLING if Rule 1 is NOT met):
    {chat_history}

    User Question:
    {question}

    Answer (Follow Rule 1 first, then Rule 2 if applicable):
    """
    custom_prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.2,
            convert_system_message_to_human=True
        )
        st.info("ℹ️ ChatGoogleGenerativeAI model (gemini-1.5-flash-latest) initialized.")
    except Exception as e:
        st.error(f"🚨 Error initializing ChatGoogleGenerativeAI: {e}")
        return None

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
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
st.set_page_config(page_title="NxtWave AI Agent", page_icon="🌊", layout="wide")
st.title("🌊 NxtWave AI Onboarding Assistant")
st.caption("I'm here to help with your NxtWave onboarding and loan process questions! Just ask.")
st.markdown("---")

if os.path.exists(DOCX_PATH):
    st.markdown(f"**Note:** This assistant uses the document: `{os.path.basename(DOCX_PATH)}` as its knowledge source. Responses are grounded in this document.")
else:
    st.error(f"🚨 **Critical Error:** Document `{DOCX_PATH}` not found. The assistant cannot function without its knowledge source.")
    st.stop()

st.markdown("The vector store is built using FAISS with embeddings from Google's models.")

example_questions = [
    "How was my child selected for this NxtWave program?",
    "What is this NxtWave course about, and how is it different from college?",
    "What specific skills will my child learn?",
    "Explain the 'Individual Development Plan' (IDP).",
    "When can my child get an internship?",
    "How does NxtWave support fast or slow learners?",
    "What are the payment options for the course fee?",
    "Can I pay in no-cost EMIs?",
    "What is an NBFC, and why use them for payments?",
    "Benefits of NBFC loan (interest-free, no collateral)?",
    "Which NBFCs do you partner with? Are they RBI-approved?",
    "Why is a co-applicant needed for EMI?",
    "What documents are needed for the loan?",
    "What should the co-applicant say in the consent video?",
    "I'm hesitant about loans. How is this EMI different?",
    "Why is auto-debit/OTP needed for EMI?",
    "What if NxtWave closes down? Is my investment secure?",
    "Any benefit to completing onboarding in 2 days?"
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
            st.rerun()

qa_chain = load_chain()

if not qa_chain:
    st.error("🚨 AI Chain could not be loaded. Please check document path, API key, document content, and console logs for errors. The application cannot proceed.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query_to_process_this_run = None
if st.session_state.user_query_from_sidebar:
    query_to_process_this_run = st.session_state.user_query_from_sidebar
    st.session_state.user_query_from_sidebar = None

typed_query = st.chat_input("Ask your question here...")
if typed_query:
    query_to_process_this_run = typed_query

if query_to_process_this_run:
    st.session_state.messages.append({"role": "user", "content": query_to_process_this_run})
    with st.chat_message("user"):
        st.markdown(query_to_process_this_run)

    try:
        with st.spinner("🤖 Thinking with Gemini... Please wait."):
            response = qa_chain.invoke({"question": query_to_process_this_run})
        answer = response["answer"]
        source_documents = response.get("source_documents", [])

        with st.chat_message("assistant"):
            st.markdown(answer)

            if source_documents and len(answer) > 50:
                with st.expander("See sources retrieved (from FAISS with Google Embeddings)"):
                    for i, doc in enumerate(source_documents):
                        section = doc.metadata.get('section', 'General')
                        st.markdown(f"**Source {i + 1} (Section: {section}):**")
                        st.caption(f"```{doc.page_content[:300]}...```")
        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        error_message = f"🚨 Error during QA chain execution with Gemini: {e}"
        st.error(error_message)
        with st.chat_message("assistant"):
            st.markdown(f"Sorry, I encountered an error processing your request: {error_message}")
        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {error_message}"})