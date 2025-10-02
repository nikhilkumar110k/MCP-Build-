# 🧠 MCP (Model Context Protocol) Project

An implementation of **MCP (Model Context Protocol)** with a custom **MCP Server + Client ecosystem**.  
This project integrates dataset retrieval, Google Search, fine-tuned Transformer models, and persistent memory into a unified context-driven system.

---

## 📌 Overview

- **MCP Server**: Hosts multiple tools (dataset retrievers, search integration, fine-tuned model prompt).  
- **MCP Client**: Implemented with **FastMCP + Python**, supports streamable HTTP responses for real-time output.  
- **Fine-Tuned Model**: **FLAN-T5 Small (R5 Transformer)** fine-tuned on a **medical dataset using LoRA**.  
- **Memory Module**: Stores past queries and supports contextual retrieval for ongoing sessions.  
- **Integrated Prompting**: Combines dataset lookups with **Google Search (SerpAPI)** and feeds results to the model.  

---

## 🛠 Tech Stack & Components

### 🔹 Core
- **Python** → Main programming language.
- **FastMCP** → Framework for building MCP Clients.
- **MCP Server** → Implements custom tools & prompts.

### 🔹 Model
- **FLAN-T5 Small (R5 Transformer)** → Base model from Google.
- **LoRA Fine-Tuning** → Efficient parameter tuning on a medical dataset.
- **Hugging Face Transformers** → Model training and inference.

### 🔹 Tools
- **Dataset Retriever** → Fetches data from internal sources.
- **Google Search (SerpAPI)** → Retrieves real-time external knowledge.
- **Custom Prompt Tool** → Feeds dataset + search results into model for reasoning.

### 🔹 Infrastructure
- **Streamable HTTP** → Enables real-time client-server communication.
- **Memory Layer** → Saves and retrieves past user queries (context continuity).
- **MCP.tools** → Encapsulates tool definitions for modular expansion.

