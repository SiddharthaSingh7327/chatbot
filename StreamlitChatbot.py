import os
import streamlit as st
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

class EnhancedDocumentQAChatbot:
    def __init__(self, api_key: str):
        """Initialize the chatbot with Gemini API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.document_content = ""
        self.document_name = ""
        self.document_images = []
        self.chat_history = []
        self.graph_rag_instance = None
        self.working_dir = None
    
    def load_document_from_bytes(self, file_bytes, file_name: str) -> bool:
        """Load a document from bytes and initialize GraphRAG."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name
            
            result = self.load_document(tmp_path)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            if result:
                self.document_name = file_name
            return result
        
        except Exception as e:
            st.error(f"Error loading document: {e}")
            return False
        
    def load_document(self, file_path: str) -> bool:
        """Load a document from file path and initialize GraphRAG."""
        try:
            path = Path(file_path)
            if not path.exists():
                st.error(f"File not found: {file_path}")
                return False
            
            self.document_content = ""
            self.document_images = []

            # Clean up previous working directory
            if self.working_dir and os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)

            self.working_dir = f"./graphrag_data_{path.stem}_{id(self)}"

            if path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
                with open(path, 'r', encoding='utf-8') as file:
                    self.document_content = file.read()
            elif path.suffix.lower() == '.pdf':
                text_content, images = self._extract_pdf_content_with_images(path)
                self.document_content = text_content
                self.document_images = images
                if not self.document_content and not self.document_images:
                    st.error("Could not extract content from PDF file.")
                    return False
            elif path.suffix.lower() in ['.docx', '.doc']:
                if path.suffix.lower() == '.docx':
                    self.document_content = self._extract_docx_text(path)
                else:
                    st.error(".doc files are not supported. Please convert to .docx format.")
                    return False
                if not self.document_content:
                    st.error("Could not extract text from Word document.")
                    return False
            elif path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
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
            
            if self.document_content:
                self._initialize_and_build_graphrag()
            return True
        except Exception as e:
            st.error(f"Error loading document: {e}")
            return False
    
    def _initialize_and_build_graphrag(self):
        """Initialize GraphRAG and build the knowledge graph."""
        try:
            with st.spinner("Building knowledge graph..."):
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
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page_text
                
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(io.BytesIO(img_data))
                            images.append(pil_image)
                        pix = None
                    except Exception as e:
                        st.warning(f"Could not extract image {img_index} from page {page_num + 1}: {e}")
            
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
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text
                    except Exception as e:
                        st.warning(f"Could not extract text from page {page_num + 1}: {e}")
                
                if not text.strip():
                    st.warning("No text could be extracted from the PDF.")
                    return ""
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def _extract_docx_text(self, docx_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = DocxDocument(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text += cell.text + " "
                    text += "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    
    def ask_question(self, question: str) -> str:
        """Ask a question using the appropriate engine (GraphRAG or Gemini)."""
        if not self.document_content and not self.document_images:
            return "No document loaded. Please load a document first."

        if self.graph_rag_instance and not self.document_images:
            try:
                with st.spinner("Analyzing with GraphRAG..."):
                    response = self.graph_rag_instance.query(question)
                    answer = response.response
                    self.chat_history.append({"question": question, "answer": answer})
                    return answer
            except Exception as e:
                return f"Error querying with GraphRAG: {e}"

        try:
            with st.spinner("Analyzing with Gemini..."):
                content_parts = []
                
                if self.document_content:
                    text_prompt = f"""
Document Name: {self.document_name}
Document Content:
---
{self.document_content}
---
"""
                    content_parts.append(text_prompt)

                if self.document_images:
                    content_parts.append("The document also contains the following images:")
                    for i, image in enumerate(self.document_images):
                        content_parts.append(f"Image {i+1}:")
                        content_parts.append(image)

                if self.chat_history:
                    content_parts.append(f"\nPrevious conversation:\n{self._format_chat_history()}")

                question_prompt = f"""
Question: {question}
Instructions:
- Analyze both text content AND any visual elements (images, charts, etc.).
- Provide a detailed and accurate answer based on ALL available content.
"""
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

def main():
    st.set_page_config(
        page_title="Enhanced Document Q&A Chatbot",
        page_icon="",
        layout="wide"
    )

    st.title("Enhanced Document Q&A Chatbot")
    st.markdown("*Powered by GraphRAG and Gemini AI*")

    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for configuration and document management
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Enter your Google Gemini API key"
        )
        
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            help="Required for GraphRAG functionality"
        )

        if st.button("Initialize Chatbot"):
            if gemini_api_key and openai_api_key:
                try:
                    os.environ["OPENAI_API_KEY"] = openai_api_key
                    st.session_state.chatbot = EnhancedDocumentQAChatbot(gemini_api_key)
                    st.success("Chatbot initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing chatbot: {e}")
            else:
                st.error("Please provide both API keys")

        st.divider()

        # Document Upload
        st.header("Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'md', 'py', 'js', 'html', 'css', 'json', 'pdf', 'docx', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'],
            help="Supported formats: Text, PDF, DOCX, Images"
        )

        if uploaded_file and st.session_state.chatbot:
            if st.button("Load Document"):
                file_bytes = uploaded_file.read()
                success = st.session_state.chatbot.load_document_from_bytes(file_bytes, uploaded_file.name)
                if success:
                    st.success(f"Document '{uploaded_file.name}' loaded successfully!")
                    st.session_state.chat_history = []
                else:
                    st.error("Failed to load document")

        # Document Info
        if st.session_state.chatbot and (st.session_state.chatbot.document_content or st.session_state.chatbot.document_images):
            st.divider()
            st.header("Document Info")
            
            with st.expander("Document Details"):
                st.write(f"**Name:** {st.session_state.chatbot.document_name}")
                if st.session_state.chatbot.document_content:
                    st.write(f"**Text Size:** {len(st.session_state.chatbot.document_content)} characters")
                    preview = st.session_state.chatbot.document_content[:200]
                    if len(st.session_state.chatbot.document_content) > 200:
                        preview += "..."
                    st.text_area("Text Preview", preview, height=100, disabled=True)
                
                if st.session_state.chatbot.document_images:
                    st.write(f"**Images:** {len(st.session_state.chatbot.document_images)} extracted")
                    for i, img in enumerate(st.session_state.chatbot.document_images):
                        st.image(img, caption=f"Image {i+1}", use_container_width=True)

            # Controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Regenerate Graph"):
                    st.session_state.chatbot.regenerate_graph()
            with col2:
                if st.button("Clear History"):
                    st.session_state.chatbot.clear_history()
                    st.session_state.chat_history = []
                    st.success("Chat history cleared!")

    # Main chat interface
    if not st.session_state.chatbot:
        st.info("Please configure your API keys and initialize the chatbot in the sidebar to get started.")
        st.markdown("""
        ### Supported File Formats:
        - **Text files:** .txt, .md, .py, .js, .html, .css, .json
        - **PDF files:** .pdf
        - **Word documents:** .docx
        - **Image files:** .jpg, .jpeg, .png, .gif, .bmp, .tiff

        ### Features:
        - **GraphRAG:** Advanced knowledge graph analysis for text documents
        - **Multimodal:** Support for images and visual content via Gemini AI
        - **Chat History:** Maintains conversation context
        - **Multiple Formats:** Wide range of supported document types
        """)
        return

    if not (st.session_state.chatbot.document_content or st.session_state.chatbot.document_images):
        st.info("👈 Please upload a document in the sidebar to start asking questions.")
        return

    # Chat interface
    st.header("Chat with Your Document")

    # Display chat history
    for i, entry in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**You:** {entry['question']}")
            st.markdown(f"**Assistant:** {entry['answer']}")
            st.divider()

    # Question input
    question = st.text_input("Ask a question about your document:", key="question_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("Ask", type="primary")
    
    if ask_button and question:
        answer = st.session_state.chatbot.ask_question(question)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })
        
        # Clear the input and rerun to show the new message
        st.rerun()

    # Suggested questions
    if st.session_state.chatbot.document_content and not st.session_state.chat_history:
        st.markdown("### Suggested Questions:")
        suggestions = [
            "Summarize the main points of this document",
            "What are the key entities mentioned?",
            "What is this document about?",
            "Extract the most important information"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    answer = st.session_state.chatbot.ask_question(suggestion)
                    st.session_state.chat_history.append({
                        "question": suggestion,
                        "answer": answer
                    })
                    st.rerun()

if __name__ == "__main__":
    main()