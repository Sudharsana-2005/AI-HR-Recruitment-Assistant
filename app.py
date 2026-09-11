import json
import os
from pathlib import Path
from typing import List

import streamlit as st
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "chroma_db"

st.set_page_config(
    page_title="HireMate AI - HR Recruitment Assistant",
    page_icon="🤖",
    layout="wide",
)

# ---------- Data helpers ----------

def load_json(filename: str):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(filename: str, data):
    with open(DATA_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# ---------- RAG ----------

@st.cache_resource(show_spinner=False)
def get_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

    documents: List[Document] = []
    for path in KNOWLEDGE_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": path.name}))

    chunks = splitter.split_documents(documents)

    vector_store = Chroma(
        collection_name="hiremate_hr_knowledge",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR),
    )

    # Add documents only when the collection is empty.
    try:
        if vector_store._collection.count() == 0:
            vector_store.add_documents(chunks)
    except Exception:
        vector_store.add_documents(chunks)

    return vector_store

# ---------- Recruitment tools ----------

@tool
def search_knowledge_base(query: str) -> str:
    """Search HR policies, job requirements, screening rules, and interview guidance."""
    store = get_vector_store()
    docs = store.similarity_search(query, k=4)
    if not docs:
        return "No relevant HR knowledge was found."
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

@tool
def screen_resume(resume_text: str, job_title: str) -> str:
    """Screen a resume against a job title using simple skill/keyword matching."""
    jobs = load_json("jobs.json")
    job = next((j for j in jobs if j["title"].lower() == job_title.lower()), None)
    if not job:
        return f"Job '{job_title}' was not found."

    resume_lower = resume_text.lower()
    matched = [s for s in job["required_skills"] if s.lower() in resume_lower]
    missing = [s for s in job["required_skills"] if s.lower() not in resume_lower]
    score = round((len(matched) / max(len(job["required_skills"]), 1)) * 100)

    return json.dumps({
        "job_title": job["title"],
        "match_score_percent": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "recommendation": "Strong match" if score >= 70 else "Needs review",
    }, indent=2)

@tool
def match_candidate(job_title: str) -> str:
    """Rank stored candidates against a job description."""
    jobs = load_json("jobs.json")
    candidates = load_json("candidates.json")
    job = next((j for j in jobs if j["title"].lower() == job_title.lower()), None)
    if not job:
        return f"Job '{job_title}' was not found."

    results = []
    required = [s.lower() for s in job["required_skills"]]

    for c in candidates:
        candidate_skills = [s.lower() for s in c.get("skills", [])]
        overlap = [s for s in required if s in candidate_skills]
        score = round((len(overlap) / max(len(required), 1)) * 100)
        results.append({
            "candidate_id": c["candidate_id"],
            "name": c["name"],
            "score_percent": score,
            "matched_skills": overlap,
            "experience_years": c.get("experience_years", 0),
        })

    results.sort(key=lambda x: (x["score_percent"], x["experience_years"]), reverse=True)
    return json.dumps(results, indent=2)

@tool
def get_candidate_information(candidate_id: str) -> str:
    """Return stored candidate profile information."""
    candidates = load_json("candidates.json")
    candidate = next(
        (c for c in candidates if c["candidate_id"].lower() == candidate_id.lower()),
        None,
    )
    return json.dumps(candidate, indent=2) if candidate else "Candidate not found."

@tool
def update_application_status(candidate_id: str, new_status: str) -> str:
    """Update a candidate's application status."""
    candidates = load_json("candidates.json")
    for candidate in candidates:
        if candidate["candidate_id"].lower() == candidate_id.lower():
            candidate["application_status"] = new_status
            save_json("candidates.json", candidates)
            return f"{candidate['name']} ({candidate_id}) status updated to {new_status}."
    return "Candidate not found."

@tool
def generate_interview_questions(job_title: str, count: int = 5) -> str:
    """Generate interview questions using the local AI model."""
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.3)
    prompt = (
        f"Generate {count} professional interview questions for the role '{job_title}'. "
        "Mix technical, behavioral, and role-specific questions. Return a numbered list."
    )
    response = llm.invoke(prompt)
    return response.content

TOOLS = [
    search_knowledge_base,
    screen_resume,
    match_candidate,
    get_candidate_information,
    update_application_status,
    generate_interview_questions,
]

SYSTEM_PROMPT = """
You are HireMate AI, an HR Recruitment Assistant.
Your job is to help recruiters with resume screening, candidate matching,
HR policy lookup, candidate information, interview question generation,
and application status updates.

Rules:
1. Use search_knowledge_base for HR policy, screening, or recruitment guidance.
2. Use recruitment tools when an action or candidate lookup is requested.
3. Do not invent candidate information or policy details.
4. Be clear that the system supports recruiters and does not make final hiring decisions.
5. Keep responses professional and concise.
"""

@st.cache_resource(show_spinner=False)
def get_agent():
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.2)
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)

# ---------- UI ----------

st.title("🤖 HireMate AI — HR Recruitment Assistant")
st.caption("Agentic AI | Agent + Tools + RAG + Memory")

with st.sidebar:
    st.header("System")
    st.write("**LLM:** Qwen 2.5 3B via Ollama")
    st.write("**Embeddings:** Nomic Embed Text")
    st.write("**Vector DB:** ChromaDB")
    st.write("**UI:** Streamlit")
    st.divider()

    st.subheader("Resume Screening")
    uploaded = st.file_uploader("Upload resume PDF", type=["pdf"])
    job_title = st.text_input("Job title", value="Python Developer")
    if uploaded:
        resume_text = extract_pdf_text(uploaded)
        st.text_area("Extracted resume text", resume_text, height=180)
        if st.button("Screen Resume"):
            result = screen_resume.invoke({
                "resume_text": resume_text,
                "job_title": job_title,
            })
            st.json(json.loads(result))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Ask: Screen my resume, match candidates, check a candidate, or generate interview questions..."
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    agent = get_agent()
    history = [
        ("human" if m["role"] == "user" else "ai", m["content"])
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("HireMate AI is working..."):
            result = agent.invoke({"messages": history})
            answer = result["messages"][-1].content
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
