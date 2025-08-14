import os
import sys
from pathlib import Path
import google.generativeai as genai
from typing import Optional
import json
import PyPDF2
from docx import Document as DocxDocument
import io
from PIL import Image
import fitz
from fast_graphrag import GraphRAG
import shutil
import tempfile
import streamlit as st

# --- The Original Chatbot Class (with print statements adapted for Streamlit) ---

class EnhancedDocumentQAChatbot:
    def __init__(self, api_key: str):
        """Initialize the chatbot with Gemini API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro') # Using 1.5 Pro for better performance
        self.document_content = ""
        self.document_name = ""
        self.document_images = []
        self.chat_history = []
        self.graph_rag_instance = None
        self.working_dir = None

    def load_document(self, file_path: str) -> bool:
        """Load a document from file path and initialize GraphRAG."""
        try:
            path = Path(file_path)
            if not path.exists():
                st.error(f"File not found: {file_path}")
                return False
            self.document_content = ""
            self.document_images = []
            if self.working_dir and os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
            self.working_dir = f"./graphrag_data_{path.stem}"
            
            # File type handling
            suffix = path.suffix.lower()
            if suffix in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
                with open(path, 'r', encoding='utf-8') as file:
                    self.document_content = file.read()
            elif suffix == '.pdf':
                text_content, images = self._extract_pdf_content_with_images(path)
                self.document_content = text_content
                self.document_images = images
                if not self.document_content and not self.document_images:
                    st.warning("Could not extract content from PDF file.")
                    return False
            elif suffix in ['.docx', '.doc']:
                if suffix == '.docx':
                    self.document_content = self._extract_docx_text(path)
                else:
                    st.error(".doc files are not supported. Please convert to .docx format.")
                    return False
                if not self.document_content:
                    st.warning("Could not extract text from Word document.")
                    return False
            elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                image = Image.open(path)
                self.document_images = [image]
                self.document_content = f"Image file: {path.name}"
            else:
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        self.document_content = file.read()
                except UnicodeDecodeError:
                    st.error(f"Cannot read file {file_path}. Unsupported file format.")
                    return False
            
            self.document_name = path.name
            st.success(f"Document '{self.document_name}' loaded successfully!")
            
            if self.document_content:
                self._initialize_and_build_graphrag()
            return True
        except Exception as e:
            st.error(f"Error loading document: {e}")
            return False

    def _initialize_and_build_graphrag(self):
        """Initializes GraphRAG and builds the knowledge graph."""
        with st.spinner("Initializing and building knowledge graph..."):
            try:
                DOMAIN = "This document contains various information. Extract key entities and their relationships."
                EXAMPLE_QUERIES = [
                    "Summarize the main points of the document.",
                    "What are the key entities mentioned?",
                    "Describe the relationships between the main entities."
                ]
                ENTITY_TYPES = ["Person", "Organization", "Location", "Date", "Topic", "Concept"]
                self.graph_rag_instance = GraphRAG(
                    working_dir=self.working_dir,
                    domain=DOMAIN,
                    example_queries="\n".join(EXAMPLE_QUERIES),
                    entity_types=ENTITY_TYPES
                )
                self.graph_rag_instance.insert(self.document_content)
                st.success("Knowledge graph built successfully!")
            except Exception as e:
                st.error(f"Error initializing or building GraphRAG: {e}")
                self.graph_rag_instance = None

    def regenerate_graph(self):
        """Forces regeneration of the knowledge graph."""
        if not self.document_content:
            st.warning("No document loaded to regenerate the graph for.")
            return
        if self.working_dir and os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)
        self._initialize_and_build_graphrag()

    def _extract_pdf_content_with_images(self, pdf_path: Path) -> tuple:
        """Extract both text and images from PDF using PyMuPDF."""
        try:
            text_content = ""
            images = []
            pdf_document = fitz.open(str(pdf_path))
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        images.append(pil_image)
                    except Exception as e:
                        st.warning(f"Could not extract image {img_index + 1} from page {page_num + 1}: {e}")
            pdf_document.close()
            return text_content.strip(), images
        except Exception as e:
            st.error(f"Error reading PDF with PyMuPDF: {e}")
            fallback_text = self._extract_pdf_text(pdf_path)
            return fallback_text, []

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Fallback method to extract text from PDF using PyPDF2."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        st.warning(f"Could not extract text from page {page_num + 1}: {e}")
            if not text.strip():
                st.warning("No text could be extracted from the PDF using the fallback method.")
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF with PyPDF2: {e}")
            return ""

    def _extract_docx_text(self, docx_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = DocxDocument(docx_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            # You can add more complex table extraction if needed
            return text.strip()
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""

    def ask_question(self, question: str) -> str:
        """Ask a question using the appropriate engine (GraphRAG or Gemini)."""
        if not self.document_content and not self.document_images:
            return "No document loaded. Please load a document first."
        
        if self.graph_rag_instance and not self.document_images:
            st.info("Using fast-graphrag for a text-based query...")
            try:
                response = self.graph_rag_instance.query(question)
                answer = response.response
                self.chat_history.append({"question": question, "answer": answer})
                return answer
            except Exception as e:
                return f"Error querying with GraphRAG: {e}"

        st.info("Using Gemini API for multimodal query...")
        try:
            content_parts = []
            if self.document_content:
                text_prompt = f"Document Name: {self.document_name}\nDocument Content:\n---\n{self.document_content}\n---"
                content_parts.append(text_prompt)
            if self.document_images:
                content_parts.append("The document also contains the following images:")
                for i, image in enumerate(self.document_images):
                    content_parts.append(f"Image {i+1}:")
                    content_parts.append(image)
            
            if self.chat_history:
                content_parts.append(f"\nPrevious conversation:\n{self._format_chat_history()}")
            
            question_prompt = f"Question: {question}\nInstructions: Analyze both text content AND any visual elements. Provide a detailed and accurate answer based on ALL available content."
            content_parts.append(question_prompt)
            
            response = self.model.generate_content(content_parts)
            answer = response.text
            self.chat_history.append({"question": question, "answer": answer})
            return answer
        except Exception as e:
            return f"Error generating response with Gemini: {e}"

    def _format_chat_history(self) -> str:
        """Format chat history for context."""
        if not self.chat_history:
            return "No previous conversation."
        history = ""
        for i, entry in enumerate(self.chat_history[-3:], 1):
            history += f"\nQ{i}: {entry['question']}\nA{i}: {entry['answer']}\n"
        return history

    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []

# --- Streamlit UI ---

st.set_page_config(page_title="Document Q&A Chatbot", layout="wide")
st.title("Enhanced Document Q&A with GraphRAG & Gemini")

# --- Sidebar for Configuration and Controls ---
with st.sidebar:
    st.header("Configuration")
    
    # API Key Input
    gemini_api_key = st.text_input("Gemini API Key", type="password", help="Get your key from Google AI Studio.")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Required by fast-graphrag.")

    if gemini_api_key and openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    else:
        st.warning("Please enter both API keys to proceed.")
        st.stop()

    # Initialize chatbot in session state
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = EnhancedDocumentQAChatbot(api_key=gemini_api_key)
            st.session_state.document_loaded = False
        except Exception as e:
            st.error(f"Failed to initialize Gemini API: {e}")
            st.stop()
    
    st.header("Document Loader")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=['txt', 'md', 'py', 'js', 'html', 'css', 'json', 'pdf', 'docx', 'jpg', 'jpeg', 'png']
    )

    if st.button("Load Document"):
        if uploaded_file is not None:
            # Create a temporary directory to store the uploaded file
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load the document using the chatbot instance
                with st.spinner(f"Loading and processing '{uploaded_file.name}'..."):
                    success = st.session_state.chatbot.load_document(file_path)
                    st.session_state.document_loaded = success
        else:
            st.warning("Please upload a file first.")

    if st.session_state.get("document_loaded", False):
        st.subheader("Document Information")
        chatbot = st.session_state.chatbot
        st.info(f"**Name:** {chatbot.document_name}")
        if chatbot.document_content:
            st.info(f"**Text Size:** {len(chatbot.document_content)} characters")
        if chatbot.document_images:
            st.info(f"**Images:** {len(chatbot.document_images)} found")
            for i, img in enumerate(chatbot.document_images):
                st.image(img, caption=f"Extracted Image {i+1}", use_column_width=True)

    st.header("Controls")
    if st.button("Regenerate Knowledge Graph"):
        if st.session_state.get("document_loaded", False):
            st.session_state.chatbot.regenerate_graph()
        else:
            st.warning("Please load a document first.")
            
    if st.button("Clear Chat History"):
        st.session_state.chatbot.clear_history()
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun()

# --- Main Chat Interface ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the document..."):
    if not st.session_state.get("document_loaded", False):
        st.warning("Please upload and load a document before asking questions.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.ask_question(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
