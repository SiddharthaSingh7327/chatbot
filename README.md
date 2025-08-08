# Enhanced Document Q&A Chatbot with Hybrid AI

This is an intelligent, command-line chatbot that analyzes the content of your local documents (including text and images) by building a knowledge graph for rapid text-based queries and leveraging a powerful multimodal AI for visual and complex questions.



## Overview

This chatbot provides a powerful way to "talk" to your documents. It uses a sophisticated hybrid approach to deliver the best possible answers:

1.  **`fast-graphrag` for Text Analysis**: For any text in your document, it builds a local knowledge graph. This allows for extremely fast and contextually-aware answers to text-based questions, powered by OpenAI models. The graphs are cached, so you only pay the processing cost once per document.
2.  **Gemini 1.5 Pro for Multimodal Analysis**: For documents containing images, charts, and diagrams, or for more complex reasoning tasks, the chatbot seamlessly switches to Google's Gemini model. This allows it to understand and answer questions about visual content.

This combination ensures you get fast, cost-effective answers for text queries and deep, multimodal understanding when you need it.

---

## Key Features

-   **Multi-Format Document Support**: Load and analyze `.pdf`, `.docx`, `.txt`, `.md`, and various image files.
-   **Hybrid AI Engine**: Automatically switches between a local knowledge graph (`fast-graphrag`) and a powerful cloud-based vision model (Gemini) for optimal performance.
-   **Knowledge Graph Caching**: Automatically saves processed document graphs to reduce API costs and speed up subsequent analysis.
-   **Interactive Command-Line Interface**: Easy-to-use commands for loading documents, managing the session, and getting help.
-   **Secure API Key Handling**: Prompts for API keys if they are not set as environment variables.

---

## Setup and Installation

Follow these steps to get the chatbot running on your local machine.

### 1. Prerequisites

-   Python 3.8 or higher

### 3. Set Up API Keys

The chatbot requires API keys for both Google Gemini and OpenAI.

**Recommended Method (Environment Variables):**

Set the following environment variables in your terminal:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

**Alternative Method (Interactive Prompt):**

If you don't set the environment variables, the program will prompt you to enter the keys directly in the terminal when you run it for the first time.

---

## How to Use

1.  **Run the Chatbot:**
    Open your terminal, navigate to the project directory, and run the script:

    ```bash
    python enhanced_chatbot.py
    ```

2.  **Load a Document:**
    Use the `/load` command followed by the path to your file.

    ```
    🤖 Ask me anything: /load path/to/your/document.pdf
    ```
    The bot will process the document and build the knowledge graph, which may take a moment for large files.

3.  **Ask Questions:**
    Once the document is loaded, simply type your question and press Enter.

    ```
    🤖 Ask me anything: What is the main conclusion of this report?
    ```
    Or, for a document with images:
    ```
    🤖 Ask me anything: Based on the chart on page 5, what was the revenue trend?
    ```
