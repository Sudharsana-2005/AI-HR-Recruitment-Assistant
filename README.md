# HireMate AI — HR Recruitment Assistant

A beginner-friendly Agentic AI project for an HR Recruitment Assistant.

## Features

- Resume PDF text extraction
- AI-powered resume screening
- Candidate-to-job matching
- Candidate profile lookup
- Interview question generation
- Application status updates
- RAG-based HR knowledge search
- Streamlit web interface
- Local Qwen 2.5 3B model with Ollama
- ChromaDB vector store

## Project Architecture

Recruiter → Streamlit UI → LangGraph Agent → RAG / Tools → Final Response

Core capabilities:
- Agent
- Tools
- RAG
- Conversation context

## Requirements

Install Python 3.10 or newer.

Install Ollama, then pull the models:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Example Prompts

Try:

```text
Match candidates for Python Developer.
```

```text
Show candidate CAND-1001.
```

```text
Generate interview questions for Python Developer.
```

```text
What does the HR screening policy say?
```

```text
Update CAND-1001 to Shortlisted.
```

## GitHub Upload

Create a GitHub repository named:

`AI-HR-Recruitment-Assistant`

Then upload the project files and folders.

Do not upload the generated `chroma_db` folder if it is large. It can be recreated automatically when the application starts.

## Academic Project Note

This project is designed as a demonstration/academic HR assistant. It supports recruiters but should not be used as the sole basis for final hiring decisions.
