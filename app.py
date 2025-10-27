# from langchain_openai import ChatOpenAI
# import os
# from crewai_tools import PDFSearchTool
# from langchain_community.tools.tavily_search import TavilySearchResults
# from crewai_tools  import tool
# from crewai import Crew
# from crewai import Task
# from crewai import Agent

# import os

# # Set the API key
# os.environ['GROQ_API_KEY'] = 'Add Your Groq API Key'

# llm = ChatOpenAI(
#     openai_api_base="https://api.groq.com/openai/v1",
#     openai_api_key=os.environ['GROQ_API_KEY'],
#     model_name="llama3-8b-8192",
#     temperature=0.1,
#     max_tokens=1000,
# )


# import requests

# pdf_url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
# response = requests.get(pdf_url)

# with open('attenstion_is_all_you_need.pdf', 'wb') as file:
#     file.write(response.content)

# ag_tool = PDFSearchTool(pdf='attenstion_is_all_you_need.pdf',
#     config=dict(
#         llm=dict(
#             provider="groq", # or google, openai, anthropic, llama2, ...
#             config=dict(
#                 model="llama3-8b-8192",
#                 # temperature=0.5,
#                 # top_p=1,
#                 # stream=true,
#             ),
#         ),
#         embedder=dict(
#             provider="huggingface", # or openai, ollama, ...
#             config=dict(
#                 model="BAAI/bge-small-en-v1.5",
#                 #task_type="retrieval_document",
#                 # title="Embeddings",
#             ),
#         ),
#     )
# )


# rag_tool.run("How did self-attention mechanism evolve in large language models?")


# import os

# # Set the Tavily API key
# os.environ['TAVILY_API_KEY'] = 'Add Your Tavily API Key'

# web_search_tool = TavilySearchResults(k=3)


# web_search_tool.run("What is self-attention mechansim in large language models?")


# @tool
# def router_tool(question):
#   """Router Function"""
#   if 'self-attention' in question:
#     return 'vectorstore'
#   else:
#     return 'web_search'
  

# Router_Agent = Agent(
#     role='Router',
#     goal='Route user question to a vectorstore or web search',
#     backstory=(
#     "You are an expert at routing a user question to a vectorstore or web search."
#     "Use the vectorstore for questions on concept related to Retrieval-Augmented Generation."
#     "You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search."
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=llm,
# )

# Retriever_Agent = Agent(
#     role="Retriever",
#     goal="Use the information retrieved from the vectorstore to answer the question",
#     backstory=(
#         "You are an assistant for question-answering tasks."
#         "Use the information present in the retrieved context to answer the question."
#         "You have to provide a clear concise answer."
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=llm,
# )

# Grader_agent =  Agent(
#   role='Answer Grader',
#   goal='Filter out erroneous retrievals',
#   backstory=(
#     "You are a grader assessing relevance of a retrieved document to a user question."
#     "If the document contains keywords related to the user question, grade it as relevant."
#     "It does not need to be a stringent test.You have to make sure that the answer is relevant to the question."
#   ),
#   verbose=True,
#   allow_delegation=False,
#   llm=llm,
# )


# hallucination_grader = Agent(
#     role="Hallucination Grader",
#     goal="Filter out hallucination",
#     backstory=(
#         "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
#         "Make sure you meticulously review the answer and check if the response provided is in alignmnet with the question asked"
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=llm,
# )


# answer_grader = Agent(
#     role="Answer Grader",
#     goal="Filter out hallucination from the answer.",
#     backstory=(
#         "You are a grader assessing whether an answer is useful to resolve a question."
#         "Make sure you meticulously review the answer and check if it makes sense for the question asked"
#         "If the answer is relevant generate a clear and concise response."
#         "If the answer gnerated is not relevant then perform a websearch using 'web_search_tool'"
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=llm,
# )

# router_task = Task(
#     description=("Analyse the keywords in the question {question}"
#     "Based on the keywords decide whether it is eligible for a vectorstore search or a web search."
#     "Return a single word 'vectorstore' if it is eligible for vectorstore search."
#     "Return a single word 'websearch' if it is eligible for web search."
#     "Do not provide any other premable or explaination."
#     ),
#     expected_output=("Give a binary choice 'websearch' or 'vectorstore' based on the question"
#     "Do not provide any other premable or explaination."),
#     agent=Router_Agent,
#     tools=[router_tool],
# )


# retriever_task = Task(
#     description=("Based on the response from the router task extract information for the question {question} with the help of the respective tool."
#     "Use the web_serach_tool to retrieve information from the web in case the router task output is 'websearch'."
#     "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'."
#     ),
#     expected_output=("You should analyse the output of the 'router_task'"
#     "If the response is 'websearch' then use the web_search_tool to retrieve information from the web."
#     "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore."
#     "Return a claer and consise text as response."),
#     agent=Retriever_Agent,
#     context=[router_task],
#    #tools=[retriever_tool],
# )


# grader_task = Task(
#     description=("Based on the response from the retriever task for the quetion {question} evaluate whether the retrieved content is relevant to the question."
#     ),
#     expected_output=("Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
#     "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
#     "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
#     "Do not provide any preamble or explanations except for 'yes' or 'no'."),
#     agent=Grader_agent,
#     context=[retriever_task],
# )


# hallucination_task = Task(
#     description=("Based on the response from the grader task for the quetion {question} evaluate whether the answer is grounded in / supported by a set of facts."),
#     expected_output=("Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
#     "Respond 'yes' if the answer is in useful and contains fact about the question asked."
#     "Respond 'no' if the answer is not useful and does not contains fact about the question asked."
#     "Do not provide any preamble or explanations except for 'yes' or 'no'."),
#     agent=hallucination_grader,
#     context=[grader_task],
# )

# answer_task = Task(
#     description=("Based on the response from the hallucination task for the quetion {question} evaluate whether the answer is useful to resolve the question."
#     "If the answer is 'yes' return a clear and concise answer."
#     "If the answer is 'no' then perform a 'websearch' and return the response"),
#     expected_output=("Return a clear and concise response if the response from 'hallucination_task' is 'yes'."
#     "Perform a web search using 'web_search_tool' and return ta clear and concise response only if the response from 'hallucination_task' is 'no'."
#     "Otherwise respond as 'Sorry! unable to find a valid response'."),
#     context=[hallucination_task],
#     agent=answer_grader,
#     #tools=[answer_grader_tool],
# )

# rag_crew = Crew(
#     agents=[Router_Agent, Retriever_Agent, Grader_agent, hallucination_grader, answer_grader],
#     tasks=[router_task, retriever_task, grader_task, hallucination_task, answer_task],
#     verbose=True,

# )

# inputs ={"question":"How does self-attention mechanism help large language models?"}

# result = rag_crew.kickoff(inputs=inputs)
import streamlit as st
import os
import tempfile
import uuid
import requests
import json
from crewai import Crew, Task, Agent
from crewai.tools import tool 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # CHANGED: Using FAISS (in-memory) instead of Chroma

# --- 1. Streamlit Session State Initialization ---

def init_session_state():
    """Initializes all necessary session state variables."""
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            "openai": os.environ.get('OPENAI_API_KEY', ''), 
        }
    if 'user_name' not in st.session_state:
        st.session_state.user_name = "Researcher"
    if 'indexed_docs' not in st.session_state:
        # Stores {'id': str, 'name': str, 'path': str, 'tool': FAISSInstance}
        st.session_state.indexed_docs = []
    if 'temp_dir' not in st.session_state:
        # Create a persistent temporary directory for the session
        st.session_state.temp_dir = tempfile.mkdtemp()
    if 'max_docs' not in st.session_state:
        st.session_state.max_docs = 5
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'embedder' not in st.session_state: # NEW: To store the OpenAIEmbeddings instance
        st.session_state.embedder = None
    if 'uploaded_files_cache' not in st.session_state:
        st.session_state.uploaded_files_cache = []
    # Ensure the key for the text input is initialized
    if 'concept_input_main' not in st.session_state:
        st.session_state.concept_input_main = ""


# --- 2. CrewAI Tool Definitions ---

# A custom Pydantic model for the output of the RAG tool (currently unused by LLM but good practice)
# class RagResult(BaseModel):
#     source: str = Field(description="The name of the paper (e.g., 'Paper 1 - Attention') where the chunk was found.")
#     chunk: str = Field(description="The most relevant text chunk found in the paper.")

@tool("Multi-Document RAG Search")
def multi_rag_tool(question: str) -> str:
    """
    Searches across all currently indexed documents (In-Memory Vector Stores)
    for relevant information based on the user's question.
    """
    results = []
    indexed_docs = st.session_state.indexed_docs
    
    if not indexed_docs:
        return "No documents indexed. Please upload a PDF or enter a URL first."

    # Iterate through all available Vector Store instances (now FAISS)
    for doc in indexed_docs:
        vector_store_instance = doc['tool'] # Renamed for generic clarity
        doc_name = doc['name']
        
        try:
            # Perform similarity search on the in-memory vector store. k=4 retrieves 4 most relevant chunks.
            retrieved_docs = vector_store_instance.similarity_search(query=question, k=4)
            
            if retrieved_docs:
                # Format the results for the LLM summarizer
                for retrieved_doc in retrieved_docs:
                    # Use page content
                    content = retrieved_doc.page_content
                    # Include the page number in the source metadata for the agent
                    results.append(f"--- SOURCE: {doc_name} (Page {retrieved_doc.metadata.get('page', 'N/A')}) ---\n{content}\n--- END SOURCE ---")

        except Exception as e:
            # Print to console for debugging
            print(f"Error querying {doc_name} from FAISS/VectorStore: {e}") 
            
    if not results:
        return "Internal RAG search yielded no relevant information across all indexed documents."
    
    return "\n\n".join(results)

# --- 3. Utility Functions ---

def clean_filename(filename):
    """Cleans up filenames for better display titles."""
    name, _ = os.path.splitext(os.path.basename(filename))
    name = name.replace('_', ' ').replace('-', ' ').title()
    return ' '.join(name.split()[:3]) # Take first three words

def download_and_index_pdf(url_or_file, is_url=False):
    """Downloads or saves a PDF and indexes it using LangChain/FAISS."""
    
    openai_key = st.session_state.api_keys.get('openai')
    embedder = st.session_state.get('embedder') # Get initialized embedder
    
    if not openai_key or not embedder:
        st.error("Cannot index document: Please click 'Initialize Agents and Tools' first.")
        return
    
    if len(st.session_state.indexed_docs) >= st.session_state.max_docs:
        st.error(f"Cannot upload more than {st.session_state.max_docs} documents.")
        return

    # Use UUID to ensure file names are unique in the temp directory
    temp_filepath = os.path.join(st.session_state.temp_dir, f"{uuid.uuid4()}.pdf")
    
    try:
        # --- File Download/Save Logic ---
        if is_url:
            st.info(f"Downloading PDF from: {url_or_file}...")
            response = requests.get(url_or_file, stream=True, timeout=30)
            response.raise_for_status()
            with open(temp_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_name_candidate = url_or_file.split('/')[-1].split('?')[0]
            if len(file_name_candidate) > 5:
                file_name = clean_filename(file_name_candidate)
            else:
                file_name = "URL Document"

        else: # Handle Uploaded File
            st.info(f"Saving uploaded file: {url_or_file.name}...")
            with open(temp_filepath, "wb") as f:
                f.write(url_or_file.getbuffer())
            file_name = clean_filename(url_or_file.name)
            
        # Create a unique, descriptive name
        doc_index = len(st.session_state.indexed_docs) + 1
        display_name = f"Paper {doc_index} - {file_name}"
        
        # --- Indexing Logic using FAISS (In-Memory) ---
        with st.spinner(f"Indexing '{display_name}' with In-Memory Vector Store (FAISS)..."):
            
            # 1. Load the document
            loader = PyPDFLoader(temp_filepath)
            documents = loader.load()
            
            # 2. Split the documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=150
            )
            chunks = text_splitter.split_documents(documents)
            
            # 3. Create FAISS Vector Store
            faiss_instance = FAISS.from_documents( # CHANGED: Use FAISS instead of Chroma
                documents=chunks, 
                embedding=embedder,
                # Removed persistence arguments as FAISS is used in-memory for the session
            )
            
            st.session_state.indexed_docs.append({
                'id': str(uuid.uuid4()),
                'name': display_name,
                'path': temp_filepath,
                'tool': faiss_instance # Storing the FAISS instance
            })
            st.success(f"Successfully indexed: **{display_name}** using In-Memory Vector Store (FAISS).") # UPDATED MESSAGE
            
    except Exception as e:
        st.error(f"Failed to index document. Check URL, file format, or API key. Error: {e}")
        # Clean up temporary file if indexing failed
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

def setup_llm_and_tools():
    """Sets up the LLM and Embedder, caching them."""
    openai_key = st.session_state.api_keys['openai'] 
    
    if not openai_key:
        st.error("Please enter your OpenAI API Key to initialize the LLM and Embedder.") 
        return None
    
    try:
        # 1. Initialize OpenAI LLM
        llm = ChatOpenAI( 
            api_key=openai_key,
            model_name="gpt-4o-mini", 
            temperature=0.1
        )
        st.session_state.llm = llm
        
        # 2. Initialize OpenAI Embeddings
        embedder = OpenAIEmbeddings(
            api_key=openai_key,
            model="text-embedding-3-small"
        )
        st.session_state.embedder = embedder
        
        return llm 
        
    except Exception as e:
        st.error(f"Failed to initialize LLM/Embedder. Check your API key. Error: {e}")
        st.session_state.llm = None # Ensure it is None on failure
        st.session_state.embedder = None # Ensure embedder is also None on failure
        return None

# CALLBACK FUNCTION to reset state for the "Start New Concept" button
def reset_app_state_callback():
    """Resets chat history and clears the concept input value."""
    st.session_state.chat_history = []
    # Set the value associated with the text input widget's key to clear it
    if 'concept_input_main' in st.session_state:
        st.session_state.concept_input_main = ""

# --- 4. Main Streamlit Logic ---

def main_app():
    init_session_state()

    st.set_page_config(layout="wide", page_title="Agentic RAG Explainer")
    st.title("📚 Agentic RAG Explainer (OpenAI RAG Only)")
    
    # --- UI Layout ---
    col_upload, col_chat = st.columns([1, 2])
    
    with col_upload:
        st.header("1. Setup & Documents")
        
        # 1. User Name
        st.session_state.user_name = st.text_input("Hi! What's your name?", value=st.session_state.user_name)

        st.subheader("OpenAI API Key (Required)")
        st.session_state.api_keys['openai'] = st.text_input( 
            "OpenAI API Key (for LLM)", type="password", 
            value=st.session_state.api_keys['openai'], help="Used for all Agent reasoning and summarization and embedding."
        )
        
        # Determine if LLM is initialized for visual feedback
        is_llm_initialized = st.session_state.get('llm') is not None
        
        # Determine button text based on initialization status
        if is_llm_initialized:
            button_text = "Agents and Tools INITIALIZED"
            button_type = "primary"
        else:
            button_text = "Initialize Agents and Tools"
            button_type = "secondary"

        if st.button(button_text, key="init_button", type=button_type, disabled=is_llm_initialized):
            llm = setup_llm_and_tools()
            if llm:
                st.success("LLM and Vector Store/Embedder Initialized! Ready for RAG.") # UPDATED MESSAGE
                st.balloons()
                st.rerun() # Rerun to update button state immediately
            else:
                st.error("Initialization failed. Please check your API key.")

        # Display successful initialization status explicitly
        if is_llm_initialized:
             st.success("✅ **STATUS: LLM and Embedder are ready.**")
        else:
             st.info("Status: Waiting for initialization.")


        # 2. Document Upload/URL Submission
        st.subheader(f"2. Index Documents ({len(st.session_state.indexed_docs)}/{st.session_state.max_docs})")
        
        # Ensure embedder is initialized before allowing uploads
        is_initialized = st.session_state.get('embedder') is not None
        
        if len(st.session_state.indexed_docs) < st.session_state.max_docs and is_initialized:
            uploaded_files = st.file_uploader("Upload a PDF file (drag & drop up to 5)", type=["pdf"], accept_multiple_files=True)
            
            if uploaded_files:
                # Use a cached list of file names or IDs instead of the file object itself
                # This prevents re-indexing the same file multiple times on immediate reruns
                uploaded_file_names = [f.name for f in st.session_state.uploaded_files_cache]
                new_files = [f for f in uploaded_files if f.name not in uploaded_file_names]

                for f in new_files:
                    download_and_index_pdf(f, is_url=False)
                
                # Update cache to prevent re-indexing on immediate rerun
                st.session_state.uploaded_files_cache = uploaded_files 
                if new_files: 
                    st.rerun() # Rerun once new files are indexed
                
                
            # Handle URL submission in a separate form to prevent immediate rerun issues
            with st.form("url_form", clear_on_submit=True):
                url_input = st.text_input("Or submit a PDF URL (one at a time)", key="url_input_form")
                submitted = st.form_submit_button("Submit URL")
                if submitted and url_input:
                    download_and_index_pdf(url_input, is_url=True)
                    st.rerun()
        elif not is_initialized:
            st.info("Initialize Agents and Tools first to enable document upload.")

        else:
            st.info(f"Maximum of {st.session_state.max_docs} documents indexed. Delete files to add new ones.")
        
        st.markdown("---")
        st.subheader("Indexed Documents")
        if st.session_state.indexed_docs:
            for doc in st.session_state.indexed_docs:
                st.markdown(f"- :white_check_mark: `{doc['name']}`")
        else:
            st.info("No documents indexed yet.")
            
    # --- Right Column: Chat and Concept Input (MODIFIED) ---
    with col_chat:
        st.header(f"3. Concept Chat with {st.session_state.user_name}")
        
        # Check if the prerequisites for search are met (LLM & Docs indexed)
        is_ready_to_search_prereqs = is_llm_initialized and st.session_state.indexed_docs

        # --- DEBUG VISUALIZATION ---
        st.caption("--- DEBUG STATUS (Internal Use) ---")
        st.json({
            "is_llm_initialized": is_llm_initialized,
            "indexed_docs_count": len(st.session_state.indexed_docs),
            # Note: concept_input_main state value is often empty at render time inside a form
            "concept_input_current_state": st.session_state.concept_input_main.strip(), 
            "is_ready_to_search_prereqs": is_ready_to_search_prereqs,
            "button_disabled_status": not is_ready_to_search_prereqs
        })
        st.caption("-----------------------------------")
        # --- END DEBUG VISUALIZATION ---

        if not is_llm_initialized:
            st.warning("Prerequisite: Please **Initialize Agents and Tools** on the left first.")
        
        elif not st.session_state.indexed_docs:
             st.warning("Prerequisite: Please **Index at least one Document** first.")
        
        # --- Search Form (Only appears if LLM is initialized) ---
        if is_llm_initialized:
            with st.form("concept_search_form", clear_on_submit=True):
            
                # 1. Concept Input (Key is concept_input_main)
                concept_input = st.text_input(
                    "Enter a concept to search and explain using the indexed documents:",
                    key="concept_input_main" # Stays the same key
                )
                
                # 2. Search Button (ALWAYS ACTIVE if PREREQS are met)
                submitted = st.form_submit_button(
                    "Search Indexed Documents (RAG)", 
                    # Button is only disabled if LLM or Docs are missing (Prerequisites)
                    # We remove the check for concept_input being present.
                    disabled=not is_ready_to_search_prereqs, 
                    use_container_width=True
                )
            
            # --- Search Execution Logic (Runs on form submission) ---
            if submitted:
                
                # VALIDATION STEP: Check if the text input field actually has content now
                concept_input_value = st.session_state.concept_input_main.strip()

                if not concept_input_value:
                    st.error("Please enter a concept (e.g., 'self attention', 'transformer model') to search.")
                    # We return early here since the input was empty
                    # No need for an extra rerun as the user can now just type and click submit again
                    return

                # Re-initialize LLM to ensure it is available (safety check)
                llm = setup_llm_and_tools() 
                if not llm:
                    return

                
                # --- Simplified Agent Definitions for Execution ---
                
                retriever_agent = Agent(
                    role="Document Retriever",
                    goal="Execute the RAG search against indexed documents (FAISS) and gather raw content.",
                    backstory=(
                        "You are a dedicated search engine specializing in academic papers. You must use the "
                        "`multi_rag_tool` to search the indexed PDF documents. Your output must be the raw, "
                        "unfiltered text chunks and their source paper names, including page numbers if available."
                    ),
                    verbose=False, # Set to False to keep console clean in Streamlit
                    allow_delegation=False,
                    llm=llm,
                    # Pass the decorated function directly.
                    tools=[multi_rag_tool] 
                )

                summarizer_agent = Agent(
                    role='Concept Explainer and Summarizer',
                    goal='Analyze search results, consolidate facts, and explain the concept clearly to the user.',
                    backstory=(
                        "You are a knowledgeable tutor and researcher."
                        "Your task is to take raw, possibly redundant, RAG search results and synthesize them into a concise, easy-to-understand explanation."
                        "Format the final answer clearly, noting the source papers (e.g., 'Paper 1 - XYZ') for each key piece of information."
                        "Always be encouraging and professional."
                    ),
                    verbose=False,
                    allow_delegation=False,
                    llm=llm,
                )
                
                # Tasks for the execution path
                retriever_task = Task(
                    description=(
                        f"Execute the RAG search for the concept '{concept_input_value}' using the `multi_rag_tool`."
                        "Return the raw search results including source names and page numbers."
                    ),
                    expected_output="A block of text containing all raw, sourced information related to the concept found in the indexed PDFs.",
                    agent=retriever_agent,
                )

                summarizer_task = Task(
                    description=(
                        f"Analyze the raw search results from the previous task for the concept '{concept_input_value}'."
                        "Synthesize this information into a clear, concise explanation suitable for a fellow researcher."
                        "Crucially, identify and list all source papers (Paper 1, Paper 2, etc.) and specific page numbers used to construct the answer."
                    ),
                    expected_output=(
                        "A comprehensive, sourced explanation formatted as: "
                        "1. A clear, introductory explanation."
                        "2. Bullet points/paragraphs detailing the concepts from the retrieved information."
                        "3. A final 'Sources Used:' section listing all document names and corresponding page numbers cited."
                    ),
                    agent=summarizer_agent,
                    context=[retriever_task],
                )
                
                execution_crew = Crew(
                    agents=[retriever_agent, summarizer_agent],
                    tasks=[retriever_task, summarizer_task],
                    verbose=1 # Keep verbose=1 to see agent steps in the console
                )
                
                # Run the crew and update chat
                st.session_state.chat_history.append({"role": "user", "content": f"**Concept Search:** {concept_input_value} (Strategy: Internal RAG Only)"})
                
                with st.spinner(f"Running Agentic System via RAG..."):
                    try:
                        # Note: The 'inputs' dict needs to match the Task placeholders 
                        # We pass the same value via a different key for simplicity
                        result = execution_crew.kickoff(inputs={"concept": concept_input_value, "concept_input_main": concept_input_value})
                        st.session_state.chat_history.append({"role": "agent", "content": result})
                        # 🎉 Trigger celebration animation after successful completion
                        st.success("Concept explanation generated successfully!")
                        st.balloons()
                        
                        
                    except Exception as e:
                        st.session_state.chat_history.append({"role": "agent", "content": f"**Error:** The Crew failed to complete the task. This often happens if the API key is incorrect or the LLM output broke the expected format. Please check your **OpenAI API Key** and **Console Log** for details. Error message: `{e}`"})

                # The form handles the rerun and clearing of st.session_state.concept_input_main
                st.rerun() 


            # 3. Chat History Display
            st.subheader("Conversation")
            chat_container = st.container(height=500, border=True)

            for message in st.session_state.chat_history:
                with chat_container.chat_message(message["role"]):
                    st.markdown(message["content"])

            # 4. Follow-up Options 
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "agent":
                st.markdown("---")
                st.subheader("Next Step")

                
                # Use on_click callback to safely reset the concept_input_main key
                if st.button(
                    "Start New Concept", 
                    on_click=reset_app_state_callback, # Call the reset function
                    help="Clear the current concept and search history for a new topic."):
                    pass # Button click automatically triggers rerun after callback
                
                st.info("To perform a new RAG search, simply enter a new concept above and click 'Search Indexed Documents (RAG)'.")


# Run the main application
if __name__ == '__main__':
    main_app()
