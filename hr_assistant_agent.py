"""
hr_assistant_agent.py
Single-file HR Assistant AI Agent (Streamlit)

How to run:
1. Save this file as hr_assistant_agent.py
2. (Optional) Create a .env file with OPENAI_API_KEY=sk-...
3. Install dependencies:
   pip install -r requirements.txt
4. Run:
   streamlit run hr_assistant_agent.py

Notes:
- This file is a self-contained demonstration agent designed to be easy to run.
- Integrations to Google Calendar, Notion, Supabase etc are provided as stubs with instructions
  on where to plug in your credentials/API calls. The agent works end-to-end locally without them.
- The agent supports:
  * Natural language questions about HR policies (RAG search across uploaded PDFs / built-in docs)
  * Leave balance lookup (local CSV demo)
  * Basic attendance lookup (demo)
  * "Schedule meeting" stub (shows how to integrate Google Calendar)
  * Logs interactions to local JSON memory
  * Simple safety checks (no destructive commands)
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# 3rd-party libs
try:
    import streamlit as st
except Exception as e:
    raise RuntimeError("This app requires streamlit. Install with `pip install streamlit`") from e

# Optional - LLM & embeddings
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Optional PDF loader
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# Optional FAISS
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env if present
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Put your key here or in .env
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai.api_key = OPENAI_API_KEY

APP_DIR = Path.cwd() / "hr_agent_data"
APP_DIR.mkdir(exist_ok=True)
DOCS_DIR = APP_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)
MEMORY_PATH = APP_DIR / "memory.json"
LEAVE_CSV = APP_DIR / "leave_balance.csv"
ATTENDANCE_CSV = APP_DIR / "attendance.csv"

# --- Utility functions ----------------------------------------------------

def save_memory(entry: Dict[str, Any]):
    try:
        if MEMORY_PATH.exists():
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        else:
            data = {"events": []}
        data["events"].append(entry)
        MEMORY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print("Memory save error:", e)

def load_memory(limit: int = 20) -> List[Dict[str, Any]]:
    try:
        if MEMORY_PATH.exists():
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            return data.get("events", [])[-limit:]
    except Exception as e:
        print("Memory load error:", e)
    return []

# Simple safe calculator (no eval on raw strings)
import ast
import operator as op

SAFE_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv
}

def safe_eval(expr: str):
    """
    Evaluate simple arithmetic expressions safely using ast.
    """
    def eval_(node):
        if isinstance(node, ast.Num):  # <number>
            return node.n
        if isinstance(node, ast.BinOp):
            return SAFE_OPERATORS[type(node.op)](eval_(node.left), eval_(node.right))
        if isinstance(node, ast.UnaryOp):
            return SAFE_OPERATORS[type(node.op)](eval_(node.operand))
        raise ValueError("Unsupported expression")
    node = ast.parse(expr, mode='eval').body
    return eval_(node)

# --- Demo data creation ---------------------------------------------------

def ensure_demo_data():
    # Leave balances demo
    if not LEAVE_CSV.exists():
        df = pd.DataFrame([
            {"EmployeeID": "E001", "Name": "Alice Kumar", "Annual": 12, "Sick": 8, "Casual": 5},
            {"EmployeeID": "E002", "Name": "Raj Singh", "Annual": 7, "Sick": 10, "Casual": 3},
            {"EmployeeID": "E003", "Name": "Priya Sharma", "Annual": 15, "Sick": 5, "Casual": 6},
        ])
        df.to_csv(LEAVE_CSV, index=False)
    # Attendance demo
    if not ATTENDANCE_CSV.exists():
        df = pd.DataFrame([
            {"EmployeeID": "E001", "Date": "2025-11-20", "Status": "Present"},
            {"EmployeeID": "E001", "Date": "2025-11-21", "Status": "Present"},
            {"EmployeeID": "E002", "Date": "2025-11-20", "Status": "Absent"},
            {"EmployeeID": "E003", "Date": "2025-11-19", "Status": "Present"},
        ])
        df.to_csv(ATTENDANCE_CSV, index=False)
    # Example HR docs (small text files) if none present
    if not any(DOCS_DIR.iterdir()):
        (DOCS_DIR / "leave_policy.txt").write_text(
            "Leave Policy:\\nAnnual leave: Employees accrue 1.0 day per month. Sick leave: up to 10 days per year.\\nCarryover: Up to 5 days carry forward.\\nProcedure: Submit leave request 3 days prior.",
            encoding="utf-8"
        )
        (DOCS_DIR / "benefits_policy.txt").write_text(
            "Benefits:\\nHealth insurance: Covered for employee only. Provident fund: 12% employer contribution.\\nGratuity: Payable after 4 years.\\nReimbursements: Travel and meals with receipts.",
            encoding="utf-8"
        )
        (DOCS_DIR / "employee_handbook.txt").write_text(
            "Employee Handbook:\\nWorking hours: 9:30 - 18:30. Remote work policy: 2 remote days per week with manager approval.",
            encoding="utf-8"
        )

ensure_demo_data()

# --- RAG / Retrieval ------------------------------------------------------

class SimpleRetriever:
    """
    Builds an in-memory list of documents (from DOCS_DIR). Optionally uses OpenAI embeddings + FAISS.
    If OpenAI key and FAISS available: will create vector index (embedding dim 1536 expected).
    Otherwise does keyword scoring.
    """
    def __init__(self):
        self.docs = []  # list of (id, text)
        self._load_docs()
        self.use_embeddings = False
        self.index = None
        self.id_to_doc = {}
        if os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE and FAISS_AVAILABLE:
            # Try to build embedding index
            try:
                self.build_faiss_index()
                self.use_embeddings = True
            except Exception as e:
                print("FAISS/Embeddings build failed:", e)
                self.use_embeddings = False

    def _load_docs(self):
        self.docs = []
        for p in DOCS_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in (".txt", ".pdf"):
                if p.suffix.lower() == ".pdf" and PDF_AVAILABLE:
                    try:
                        text = ""
                        reader = PdfReader(str(p))
                        for pg in reader.pages:
                            text += pg.extract_text() or ""
                        text = text.strip()
                    except Exception:
                        text = p.read_text(encoding="utf-8", errors="ignore")
                else:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                self.docs.append((p.name, text))
        # lightweight chunking: split by paragraphs
        new_docs = []
        for name, text in self.docs:
            parts = [p.strip() for p in re.split(r"\\n{2,}|\\.\\s+", text) if p.strip()]
            for i, part in enumerate(parts):
                doc_id = f"{name}__{i}"
                new_docs.append((doc_id, part))
        self.docs = new_docs
        self.id_to_doc = {doc_id: text for doc_id, text in self.docs}

    def build_faiss_index(self):
        # OpenAI embeddings dimension for text-embedding-3-large is 3072 recently but we try 1536 common dim
        # We'll call openai.Embedding.create with model "text-embedding-3-small" or "text-embedding-3-large"
        model = "text-embedding-3-small"
        emb_dim = None
        embeddings = []
        for doc_id, txt in tqdm(self.docs, desc="Embedding docs"):
            r = openai.Embedding.create(input=txt[:3000], model=model)
            vec = r["data"][0]["embedding"]
            embeddings.append(vec)
            emb_dim = len(vec)
        if emb_dim is None:
            raise RuntimeError("No embeddings created")
        xb = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(emb_dim)
        index.add(xb)
        self.index = index
        self.id_to_idx = {i: doc_id for i, (doc_id, _) in enumerate(self.docs)}
        self.idx_to_id = {v: k for k, v in self.id_to_idx.items()}

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, str]]:
        if self.use_embeddings and self.index is not None:
            # embed query
            model = "text-embedding-3-small"
            r = openai.Embedding.create(input=query[:3000], model=model)
            qvec = np.array(r["data"][0]["embedding"]).astype("float32")
            D, I = self.index.search(np.array([qvec]), k)
            res = []
            for idx in I[0]:
                if idx < 0:
                    continue
                doc_id = self.id_to_idx[idx]
                res.append((doc_id, self.id_to_doc[doc_id]))
            return res
        else:
            # simple keyword scoring
            scores = []
            qwords = set(re.findall(r"\\w+", query.lower()))
            for doc_id, text in self.docs:
                words = set(re.findall(r"\\w+", text.lower()))
                score = len(qwords & words)
                scores.append((score, doc_id, text))
            scores.sort(reverse=True, key=lambda x: x[0])
            res = [(doc_id, text) for score, doc_id, text in scores[:k] if score > 0]
            return res

# single global retriever
RETRIEVER = SimpleRetriever()

# --- LLM Adapter ----------------------------------------------------------

def llm_chat(prompt: str, system: Optional[str] = None, max_tokens: int = 400) -> str:
    """
    If OPENAI_API_KEY present, call OpenAI ChatCompletion; otherwise use a local fallback.
    """
    CLEANED_PROMPT = prompt.strip()
    if OPENAI_API_KEY and OPENAI_AVAILABLE:
        try:
            # Use GPT-4o-like ChatCompletion if available; fallback to gpt-3.5-turbo
            model = "gpt-4o-mini" if False else "gpt-3.5-turbo"
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": CLEANED_PROMPT})
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0
            )
            text = resp["choices"][0]["message"]["content"].strip()
            return text
        except Exception as e:
            print("OpenAI call failed:", e)
            # fallback to local
    # Local heuristic fallback
    lp = CLEANED_PROMPT.lower()
    # If it looks like a question about leave/benefits -> use retriever
    if "leave" in lp or "benefit" in lp or "policy" in lp or "reimburse" in lp:
        docs = RETRIEVER.retrieve(CLEANED_PROMPT, k=3)
        if docs:
            assembled = "\\n\\n".join([f"- Source {doc_id}:\\n{txt}" for doc_id, txt in docs])
            return f"I found these policy snippets relevant to your question:\\n\\n{assembled}\\n\\n(For production, provide a clear summary. Install OpenAI key for better answers.)"
        else:
            return "I couldn't find matching policy text in the documents. Try uploading a policy PDF or ask HR."
    if "balance" in lp and re.search(r"\\be\\d{3}\\b", lp.upper()):
        # sample reply about leave
        m = re.search(r"(E\\d{3})", CLEANED_PROMPT.upper())
        emp = m.group(1) if m else "E001"
        df = pd.read_csv(LEAVE_CSV)
        row = df[df["EmployeeID"].str.upper() == emp.upper()]
        if not row.empty:
            r = row.iloc[0]
            return f"Leave balance for {r['Name']} ({r['EmployeeID']}): Annual {r['Annual']} days, Sick {r['Sick']} days, Casual {r['Casual']} days."
        return "Employee not found in local DB."
    # generic fallback
    return "I am running in offline mode (no OpenAI key). For best results, set OPENAI_API_KEY. I can: answer policy questions (if docs uploaded), check demo leave balances, or schedule meeting stubs."

# --- Tools (functions the agent can call) ---------------------------------

def tool_hr_rag_search(query: str) -> str:
    docs = RETRIEVER.retrieve(query, k=4)
    if not docs:
        return "No relevant HR documents found. You can upload PDF/TXT files to the Documents panel."
    assembled = []
    for doc_id, txt in docs:
        snippet = txt.strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        assembled.append(f"Source: {doc_id}\\n{snippet}")
    # Summarize with LLM if available
    summary_prompt = f"Summarize these HR snippets for a user question: {query}\\n\\n" + "\\n\\n".join([s for s in assembled])
    summary = llm_chat(summary_prompt, system="You are a helpful HR assistant that summarizes policies succinctly.", max_tokens=300)
    return summary

def tool_check_leave_balance(employee_id: str) -> str:
    try:
        df = pd.read_csv(LEAVE_CSV)
        row = df[df["EmployeeID"].str.upper() == employee_id.strip().upper()]
        if row.empty:
            return f"Employee {employee_id} not found in leave DB."
        r = row.iloc[0]
        return f"Leave balance for {r['Name']} ({r['EmployeeID']}): Annual {r['Annual']} days, Sick {r['Sick']} days, Casual {r['Casual']} days."
    except Exception as e:
        return f"Error reading leave DB: {e}"

def tool_get_attendance(employee_id: str, days: int = 7) -> str:
    try:
        df = pd.read_csv(ATTENDANCE_CSV)
        rows = df[df["EmployeeID"].str.upper() == employee_id.strip().upper()].sort_values(by="Date", ascending=False).head(days)
        if rows.empty:
            return f"No attendance records found for {employee_id}."
        return rows.to_string(index=False)
    except Exception as e:
        return f"Attendance lookup error: {e}"

def tool_schedule_meeting(email: str, start_iso: str, duration_minutes: int = 30) -> str:
    """
    Stub: returns a fake calendar link. To integrate with Google Calendar:
    - Create service account or OAuth credentials
    - Use google-api-python-client to insert an event
    This stub returns a simulated confirmation.
    """
    # Basic validation
    if "@" not in email:
        return "Invalid email address."
    start = start_iso
    end = start_iso  # not parsing in stub
    fake_link = f"https://calendar.google.com/calendar/r/event?eid=fake-{int(time.time())}"
    return f"Meeting scheduled with {email} at {start} for {duration_minutes} minutes. Link: {fake_link}\\n\\n(Plug Google Calendar API to make real bookings.)"

def tool_log_to_memory(note: str) -> str:
    entry = {"time": time.time(), "note": note}
    save_memory(entry)
    return "Logged to local memory."

def tool_calculator(expression: str) -> str:
    try:
        value = safe_eval(expression)
        return f"Result: {value}"
    except Exception as e:
        return f"Calculator error: {e}"

# --- Agent Orchestration (simple planner + tool execution) ---------------

SYSTEM_PROMPT = """You are HR Assistant Agent. You must:
- Answer HR policy questions using the RAG retriever if policy related.
- Use Check_Leave_Balance if user asks about leave balance (format: E###).
- Use Get_Attendance to fetch attendance.
- Use Schedule_Meeting for booking.
- If the user asks for a calculation, use the CALCULATOR tool.
- If unsure, ask a clarifying question.
Be concise (max ~200 words)."""

def plan_and_execute(user_input: str) -> Dict[str, Any]:
    """
    Very simple planner:
    - Inspect user_input for intent keywords and call an appropriate tool.
    - Fall back to RAG search + LLM summarization.
    """
    save_memory({"time": time.time(), "role": "user", "text": user_input})
    ui = user_input.strip()
    ui_lower = ui.lower()

    # 1) Leave balance intent
    m = re.search(r"(E\\d{3})", ui.upper())
    if "leave balance" in ui_lower or "leave balance" in ui_lower or m:
        emp = m.group(1) if m else None
        if not emp:
            # try to extract employee id pattern differently
            emp = None
        if emp:
            out = tool_check_leave_balance(emp)
            save_memory({"time": time.time(), "role": "agent", "text": out})
            return {"tool": "Check_Leave_Balance", "output": out}
        else:
            # Ask for employee ID
            out = "Please provide your Employee ID (format: E001) to check leave balance."
            save_memory({"time": time.time(), "role": "agent", "text": out})
            return {"tool": "ask_for_employee_id", "output": out}

    # 2) Attendance intent
    if "attendance" in ui_lower or "present" in ui_lower:
        m = re.search(r"(E\\d{3})", ui.upper())
        emp = m.group(1) if m else None
        if emp:
            out = tool_get_attendance(emp, days=7)
            save_memory({"time": time.time(), "role": "agent", "text": out})
            return {"tool": "Get_Attendance", "output": out}
        else:
            out = "Please provide the Employee ID (like E001) for attendance lookup."
            save_memory({"time": time.time(), "role": "agent", "text": out})
            return {"tool": "ask_for_employee_id", "output": out}

    # 3) Schedule meeting intent
    if "schedule" in ui_lower or "book" in ui_lower or "meeting" in ui_lower:
        # try to extract email and datetime (simple)
        emails = re.findall(r"[\w\.-]+@[\w\.-]+", ui)
        # naive date extraction (ISO-like)
        dt_match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", ui)
        if emails and dt_match:
            out = tool_schedule_meeting(emails[0], dt_match.group(0))
            save_memory({"time": time.time(), "role": "agent", "text": out})
            return {"tool": "Schedule_Meeting", "output": out}
        else:
            out = ("To schedule a meeting, provide an email and start time in ISO format.\\n"
                   "Example: 'Schedule meeting with raj@company.com at 2025-12-01T10:00'.")
            save_memory({"time": time.time(), "role": "agent", "text": out})
            return {"tool": "clarify_schedule", "output": out}

    # 4) Calculator intent
    if re.search(r"^[0-9\\.\\s\\+\\-\\*\\/\\(\\)]+$", ui) or "calculate" in ui_lower or "estimate" in ui_lower:
        # attempt to extract math expression
        expr_match = re.search(r"calculate\\s*[:\\-]?\\s*(.+)", ui_lower)
        expr = None
        if expr_match:
            expr = expr_match.group(1)
        else:
            # fallback: treat the whole input as expression
            expr = ui
        out = tool_calculator(expr)
        save_memory({"time": time.time(), "role": "agent", "text": out})
        return {"tool": "Calculator", "output": out}

    # 5) HR policy / general: use RAG + LLM
    out = tool_hr_rag_search(ui)
    save_memory({"time": time.time(), "role": "agent", "text": out})
    return {"tool": "HR_RAG_Search", "output": out}

# --- Streamlit UI ---------------------------------------------------------

st.set_page_config(page_title="HR Assistant Agent", layout="centered")
st.title("🤖 HR Assistant — Single-file Agent")
st.markdown(
    """
    This is a single-file demo HR Assistant Agent.
    - Ask HR policy questions (upload docs in Documents panel)
    - Check leave balances: type "Leave balance for E001" or just "E001 leave balance"
    - Get attendance: "Attendance E002"
    - Schedule meeting: "Schedule meeting with raj@company.com at 2025-12-01T10:00"
    - Calculator: type "calculate 1200/12" or "calculate: 23 * (4 + 1)"
    """
)

# Sidebar: documents and uploads
st.sidebar.header("Documents & Data")
st.sidebar.write("Upload HR PDFs/TXT to improve policy answers (persisted to hr_agent_data/docs).")

uploaded = st.sidebar.file_uploader("Upload .pdf or .txt (multiple allowed)", accept_multiple_files=True, type=["pdf", "txt"])
if uploaded:
    for up in uploaded:
        dest = DOCS_DIR / up.name
        with open(dest, "wb") as f:
            f.write(up.read())
    st.sidebar.success(f"Saved {len(uploaded)} file(s). Rebuilding retriever...")
    # rebuild retriever (simple approach: reinstantiate)
    RETRIEVER.__init__()  # re-load docs and rebuild if possible
    st.sidebar.info("Documents updated.")

# Show current documents
st.sidebar.markdown("**Current documents:**")
for p in sorted(DOCS_DIR.iterdir()):
    st.sidebar.write(f"- {p.name}")

# Show demo DB files
st.sidebar.markdown("**Demo DBs:**")
st.sidebar.write(f"- Leave DB: {LEAVE_CSV.name}")
st.sidebar.write(f"- Attendance DB: {ATTENDANCE_CSV.name}")

# Interaction panel
query = st.text_input("Type your question or command", placeholder="e.g. 'What is the leave policy for maternity?' or 'Leave balance E002'")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Send"):
        if not query:
            st.warning("Type something before sending.")
        else:
            with st.spinner("Agent thinking..."):
                result = plan_and_execute(query)
            st.markdown("**Agent response:**")
            st.write(result["output"])

with col2:
    if st.button("Clear memory"):
        try:
            MEMORY_PATH.unlink(missing_ok=True)
            st.success("Memory cleared.")
        except Exception as e:
            st.error(f"Could not clear memory: {e}")

# Show memory
st.subheader("Conversation / Memory (recent)")
mem = load_memory(20)
if mem:
    for e in mem[-20:]:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.get("time", time.time())))
        text = e.get("text") or e.get("note") or json.dumps(e)
        st.markdown(f"- `{t}` — {text}")
else:
    st.info("No memory yet. Interact with the agent to see stored events here.")

# Tools quick actions
st.subheader("Quick tools")

with st.expander("Check leave balance (demo)"):
    emp_in = st.text_input("Employee ID (E001 format)", value="E001", key="leave_emp")
    if st.button("Check Leave", key="btn_check_leave"):
        out = tool_check_leave_balance(emp_in)
        st.write(out)

with st.expander("Attendance lookup (demo)"):
    emp_att = st.text_input("Employee ID for attendance", value="E001", key="att_emp")
    if st.button("Get Attendance", key="btn_att"):
        out = tool_get_attendance(emp_att)
        st.text(out)

with st.expander("Schedule meeting (stub)"):
    email = st.text_input("Invitee email", value="raj@company.com", key="cal_email")
    start = st.text_input("Start (ISO) e.g. 2025-12-01T10:00", value="2025-12-01T10:00", key="cal_start")
    dur = st.number_input("Duration minutes", value=30, key="cal_dur")
    if st.button("Schedule (stub)", key="btn_sched"):
        out = tool_schedule_meeting(email, start, int(dur))
        st.write(out)

with st.expander("Upload a policy file (advanced)"):
    st.markdown("Upload PDF or TXT to add to the retriever corpus (same as left sidebar).")

st.subheader("Admin / Deployment Notes")
st.markdown(
    """
    - To enable high-quality LLM answers, set `OPENAI_API_KEY` in your environment or a `.env` file.
    - To enable vector-search with real embeddings, ensure `faiss-cpu`, `numpy` and `openai` are installed and an API key is set.
    - To integrate Google Calendar, replace `tool_schedule_meeting` with an implementation using `google-api-python-client`.
    - To log to a central DB (Notion / Supabase), replace `tool_log_to_memory` with real API calls and secure keys in env vars.
    """
)

st.markdown("---")
st.caption("This single-file demo is intentionally minimal to make it easy to run locally and extend for your AI Agent challenge. If you want, I can generate a multi-file repo with LangChain, Chroma/FAISS config, GitHub Actions, and a Dockerfile next.")
