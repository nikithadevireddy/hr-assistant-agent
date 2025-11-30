# HR Assistant AI Agent (Single-file)

This repository contains a single-file Streamlit app `hr_assistant_agent.py` — a demo HR Assistant AI Agent.
It supports:
- RAG search over uploaded HR docs (TXT/PDF).
- Demo leave balance and attendance lookups (CSV).
- Meeting scheduling stub (replace with Google Calendar integration).
- Local memory logging (JSON).

## How to run

1. Create a virtual environment and activate it.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Add your OpenAI API key in `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```
4. Start the app:
   ```
   streamlit run hr_assistant_agent.py
   ```

The app will open at http://localhost:8501

## Files
- `hr_assistant_agent.py` — main app
- `requirements.txt` — dependencies
- `.env.example` — example environment variables
- `docs/` — sample HR text documents
- `hr_agent_data/` — created at runtime to store docs and data
