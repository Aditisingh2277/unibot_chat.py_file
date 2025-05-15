import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import chromadb
from langchain.vectorstores import Chroma
import time

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="GBPUAT-UniBot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-message.user {
        background-color: #F3F4F6;
    }
    
    .chat-message.assistant {
        background-color: #EEF2FF;
    }
    
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    
    .chat-message .message {
        flex-grow: 1;
    }
    
    .source-doc {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Center the header */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
    }

    .header-container h1 {
        color: #1E3A8A;
        font-size: 2.5rem;
        margin: 0;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

CUSTOM_PROMPT = PromptTemplate(
    template="""<|system|>
    You are a university administrative assistant for GBPUAT (Govind Ballabh Pant University of Agriculture and Technology). 
    Use the provided context to answer questions thoroughly and accurately.
    
    Guidelines:
    1.give short answers
     2. For exact question matches, use the provided answer verbatim
    3. Combine information from multiple sources when needed
    4. Structure complex answers with bullet points
    5. Cite document sources when possible    
    
    
    Context: {context}
    
    Question: {question}
    
    Answer: Let me help you with that.
    """,
    input_variables=["context", "question"]
)

def load_qa_chain(db_path="./db"):
    """Initialize QA system with academic-focused settings"""
    try:
        client = chromadb.PersistentClient(path=db_path)
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = Chroma(
            client=client,
            collection_name="university_docs",
            embedding_function=embeddings,
            persist_directory=db_path
        )
        
        return RetrievalQA.from_chain_type(
            llm=Ollama(
                model="phi3:mini-128k",
                temperature=0.3,
                num_ctx=4096
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,
                    "score_threshold": 0.65
                }
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
        )
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = load_qa_chain()

# Main chat interface
st.markdown("""
<div class="header-container">
    <h1>🎓 UniBot - Query Assistant</h1>
</div>
""", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message['role']}">
            <div class="avatar">{'🧑' if message['role'] == 'user' else '🤖'}</div>
            <div class="message">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources for assistant messages
        if message['role'] == 'assistant' and 'sources' in message:
            with st.expander("📚 View Sources"):
                for idx, doc in enumerate(message['sources']):
                    st.markdown(f"""
                    **Source {idx + 1}**: {doc.metadata.get('source', 'Unknown')}
                    <div class="source-doc">
                        {doc.page_content[:400]}...
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything about GBPUAT..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    st.markdown(f"""
    <div class="chat-message user">
        <div class="avatar">🧑</div>
        <div class="message">{prompt}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get and display assistant response
    try:
        with st.spinner("🤔 Let me check that for you..."):
            if st.session_state.qa_chain is not None:
                result = st.session_state.qa_chain({"query": prompt})
                response = result['result']
                sources = result.get('source_documents', [])
                
                # Display assistant response
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="avatar">🤖</div>
                    <div class="message">{response}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
                # Show sources
                if sources:
                    with st.expander("📚 View Sources"):
                        for idx, doc in enumerate(sources):
                            st.markdown(f"""
                            **Source {idx + 1}**: {doc.metadata.get('source', 'Unknown')}
                            <div class="source-doc">
                                {doc.page_content[:400]}...
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.error("Sorry, I'm having trouble accessing the university information. Please try again later.")
    except Exception as e:
        st.error(f"I apologize, but I encountered an error: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I apologize, but I encountered an error while processing your question. Please try again."
        })