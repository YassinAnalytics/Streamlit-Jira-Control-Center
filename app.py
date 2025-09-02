# app.py — Streamlit × n8n × AI (Demo data + n8n-only option, no direct Jira code)
# Run:  python -m streamlit run app.py

import os
import json
import uuid
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# Optional: load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

# -----------------------------
# Page config + global CSS
# -----------------------------
st.set_page_config(page_title="n8n × AI — Jira Control Panel (Demo)", layout="wide")

st.markdown("""
<style>
:root {
  --brand-1: #1f77b4;
  --brand-2: #ff7f0e;
  --ok: #2ca02c;
  --warn: #ff7f0e;
  --ko: #d62728;
}
@media (prefers-color-scheme: dark) {
  :root { --brand-1:#4fb1ff; --brand-2:#ffb066; }
}
.main-header {
  background: linear-gradient(90deg, var(--brand-1), var(--brand-2));
  padding: 1rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1.2rem;
}
.metric-card {
  background: rgba(127,127,127,0.07);
  padding: 0.9rem; border-radius: 10px; border-left: 4px solid var(--brand-1);
}
.small-muted { font-size:12px; opacity:.7; }
</style>
<style>
.clear-form-btn {
    background-color: #ff4b4b;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
}
.clear-form-btn:hover {
    background-color: #ff3333;
}
</style>            
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🚀 My Portfolio — Jira Control Center (via n8n)</h1></div>', unsafe_allow_html=True)

# -----------------------------
# Constants & defaults
# -----------------------------
DATA_PATH = Path(__file__).parent / "tickets_jira_demo.json"

DEFAULTS = {
    # AI
    "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic_model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
    "mistral_model": os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
    "deepseek_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    "deepseek_base": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3"),
    "ollama_base": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
}

WIP_LIMIT = 10  # WIP warning threshold
DEFAULT_N8N_ONLY = os.getenv("N8N_ONLY", "true").lower() == "true"  # default: send to n8n and do NOT save locally

# -----------------------------
# Demo data utilities
# -----------------------------
def save_tickets(tickets: List[Dict[str, Any]]):
    with DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2, ensure_ascii=False)

def generate_demo_tickets() -> List[Dict[str, Any]]:
    """Generate 60 realistic demo tickets (Epics, Stories, Tasks, Bugs, Sub-tasks)."""
    import random
    random.seed(42)

    PROJECT_KEY = "DEMO"
    counts = {"Epic": 6, "Story": 18, "Task": 20, "Bug": 12, "Sub-task": 4}
    assert sum(counts.values()) == 60

    statuses = ["Backlog", "To Do", "In Progress", "In Review", "Blocked", "Done"]
    priorities = ["Low", "Medium", "High", "Critical"]
    components = ["Frontend", "Backend", "API", "Auth", "Payments", "Data", "Infra", "Mobile", "Analytics"]
    labels_pool = ["v1", "v2", "hotfix", "refactor", "ux", "performance", "security", "onboarding", "etl", "ml"]
    people = [
        "Alex Martin", "Sophie Bernard", "Liam Dupont", "Camille Leroy", "Noah Petit",
        "Eva Laurent", "Lucas Robert", "Manon Garcia", "Hugo Moreau", "Jeanne Fontaine",
        "Malo Caron", "Nina Durand", "Rayan Weber", "Léna Marchand", "Tom Chevalier"
    ]

    start_date = datetime(2025, 3, 1)
    end_date = datetime(2025, 8, 20)

    def rdate(a: datetime, b: datetime) -> datetime:
        import random
        delta = (b - a).days
        return a + timedelta(days=random.randint(0, delta),
                             hours=random.randint(0, 23),
                             minutes=random.randint(0, 59))

    def iso(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    tickets: List[Dict[str, Any]] = []
    next_index = 1

    # Epics
    epic_keys: List[str] = []
    for _ in range(counts["Epic"]):
        created = rdate(start_date, end_date - timedelta(days=30))
        import random
        status = random.choices(["Backlog", "To Do", "In Progress", "In Review", "Done"], weights=[1, 2, 2, 1, 3])[0]
        key = f"{PROJECT_KEY}-{next_index}"
        next_index += 1
        t = {
            "id": str(uuid.uuid4()),
            "key": key,
            "project": PROJECT_KEY,
            "type": "Epic",
            "summary": f"[Epic] {random.choice(['Onboarding','Payments','Mobile Revamp','Data Pipeline','AI Assistant','Security Hardening'])}",
            "description": "High-level initiative grouping related stories and tasks.",
            "status": status,
            "priority": random.choice(priorities),
            "assignee": random.choice(people) if random.random() < 0.7 else None,
            "reporter": random.choice(people),
            "created": iso(created),
            "updated": iso(created + timedelta(days=random.randint(0, 30))),
            "due_date": iso(created + timedelta(days=random.randint(21, 90))) if random.random() < 0.6 else None,
            "labels": random.sample(labels_pool, k=random.randint(1, 3)),
            "components": random.sample(components, k=random.randint(1, 2)),
            "sprint": None,
            "story_points": None,
            "epic_link": None,
            "parent": None,
            "attachments": [],
            "watchers": random.randint(0, 7),
            "url": f"https://jira.example.com/browse/{key}"
        }
        tickets.append(t)
        epic_keys.append(key)

    # Stories / Tasks / Bugs
    def add_items(kind: str, count: int):
        nonlocal next_index
        import random
        for _ in range(count):
            created = rdate(start_date, end_date - timedelta(days=5))
            status = random.choices(statuses, weights=[1, 3, 4, 2, 1, 3])[0]
            key = f"{PROJECT_KEY}-{next_index}"
            next_index += 1
            summary_core = random.choice([
                "Implement user login", "Refactor payment service", "Build analytics dashboard",
                "Improve API rate limiting", "Fix flaky tests in CI", "Add dark mode", "Migrate to OAuth2",
                "Optimize SQL queries", "Implement push notifications", "Enhance onboarding flow",
                "Cache layer for search", "GDPR data export", "Error tracking integration",
                "Streamlit UI polish", "AI summarizer endpoint"
            ])
            t = {
                "id": str(uuid.uuid4()),
                "key": key,
                "project": PROJECT_KEY,
                "type": kind,
                "summary": f"[{kind}] {summary_core}",
                "description": f"Auto-generated demo {kind.lower()} for the Streamlit × n8n portfolio app.",
                "status": status,
                "priority": random.choices(["Low", "Medium", "High", "Critical"], weights=[2, 4, 3, 1])[0],
                "assignee": random.choice(people) if random.random() < 0.85 else None,
                "reporter": random.choice(people),
                "created": iso(created),
                "updated": iso(created + timedelta(days=random.randint(0, 20))),
                "due_date": iso(created + timedelta(days=random.randint(7, 45))) if random.random() < 0.5 else None,
                "labels": random.sample(labels_pool, k=random.randint(1, 3)),
                "components": random.sample(components, k=random.randint(1, 3)),
                "sprint": f"Sprint {random.randint(10, 22)}" if random.random() < 0.7 else None,
                "story_points": random.choice([1, 2, 3, 5, 8, 13]) if kind == "Story" else None,
                "epic_link": random.choice(epic_keys) if random.random() < 0.7 else None,
                "parent": None,
                "attachments": [],
                "watchers": random.randint(0, 6),
                "url": f"https://jira.example.com/browse/{key}"
            }
            tickets.append(t)

    add_items("Story", 18)
    add_items("Task", 20)
    add_items("Bug", 12)

    # Sub-tasks
    parents = [t["key"] for t in tickets if t["type"] in ("Story", "Task", "Bug")]
    for _ in range(counts["Sub-task"]):
        import random
        created = rdate(start_date, end_date - timedelta(days=2))
        parent_key = parents[int(uuid.uuid4().int % len(parents))]
        key = f"{PROJECT_KEY}-{next_index}"
        next_index += 1
        t = {
            "id": str(uuid.uuid4()),
            "key": key,
            "project": PROJECT_KEY,
            "type": "Sub-task",
            "summary": f"[Sub-task] {random.choice(['Write unit tests','Update docs','Add monitoring','Create migration','Review logs'])}",
            "description": "Smaller unit of work under a parent issue.",
            "status": random.choices(["To Do", "In Progress", "In Review", "Done"], weights=[3, 4, 2, 3])[0],
            "priority": random.choices(["Low", "Medium", "High", "Critical"], weights=[3, 3, 2, 1])[0],
            "assignee": random.choice(people) if random.random() < 0.9 else None,
            "reporter": random.choice(people),
            "created": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "updated": (created + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "due_date": None,
            "labels": ["demo"],
            "components": ["API"],
            "sprint": None,
            "story_points": None,
            "epic_link": None,
            "parent": parent_key,
            "attachments": [],
            "watchers": 0,
            "url": f"https://jira.example.com/browse/{key}"
        }
        tickets.append(t)

    return tickets

def load_demo_tickets() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        st.warning("File 'tickets_jira_demo.json' not found. Click to generate 60 demo tickets.")
        if st.button("Generate demo dataset (60 tickets)"):
            tickets = generate_demo_tickets()
            save_tickets(tickets)
            st.success("Demo dataset created. Click Rerun if needed.")
        return []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def to_df(tickets: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for t in tickets:
        rows.append({
            **t,
            "labels": ", ".join(t.get("labels") or []),
            "components": ", ".join(t.get("components") or []),
        })
    return pd.DataFrame(rows)

def next_key(tickets: List[Dict[str, Any]], project: str = "DEMO") -> str:
    nums = []
    for t in tickets:
        try:
            nums.append(int(str(t.get("key", "")).split("-")[-1]))
        except Exception:
            pass
    n = max(nums) + 1 if nums else 1
    return f"{project}-{n}"

# -----------------------------
# n8n webhook
# -----------------------------
def post_to_n8n(payload: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
    """
    Returns (status, body). Status is 'no_url', 'error: ...', or HTTP status code string.
    Body is parsed JSON if possible, else text, else None.
    """
    url = st.session_state.get("n8n_url") or os.getenv("N8N_WEBHOOK_URL")
    if not url:
        return "no_url", None
    

    # Log of payload before sending (for debug)
    print("=== PAYLOAD STREAMLIT ===")
    print("URL:", url)
    print("Payload:", json.dumps(payload, indent=2, default=str))
    print("=========================")

    try:
        r = requests.post(url, json=payload, timeout=15)
        print(f"Response status: {r.status_code}")
        print(f"Response text: {r.text}")
        try:
            body = r.json()
        except Exception:
            body = r.text
        return str(r.status_code), body
    except Exception as e:
        print(f"Erreur requête: {e}")
        return f"error: {e}", None


# AI — multi-provider summaries
# -----------------------------
def summarize_locally(df: pd.DataFrame) -> List[str]:
    total = len(df)
    by_status = df["status"].value_counts().to_dict() if "status" in df else {}
    by_type = df["type"].value_counts().to_dict() if "type" in df else {}
    top_assignees = (
        df["assignee"].value_counts().head(3).index.tolist()
        if "assignee" in df and not df["assignee"].isna().all()
        else []
    )
    bullets = [
        f"{total} filtered tickets (status mix: " + ", ".join(f"{k}={v}" for k, v in by_status.items()) + ").",
        "By type: " + ", ".join(f"{k}={v}" for k, v in by_type.items()),
        "Top assignees: " + (", ".join(top_assignees) if top_assignees else "—"),
        "Risks: presence of 'Blocked' or long 'In Review' items.",
        "Next steps: unblock 'Blocked', prioritize High/Critical, finish reviews.",
    ]
    return bullets

def build_summary_prompt(df: pd.DataFrame, lang: str = "English") -> str:
    cols = [c for c in ["key", "type", "status", "priority", "assignee", "summary"] if c in df.columns]
    sample = df[cols].head(50).to_dict(orient="records")
    return (
        f"You are a PM assistant. Summarize the following Jira tickets in 5 bullets: "
        f"context, problem, impact, risks, next steps. Be concise. Write in {lang}.\n"
        f"JSON sample:\n{json.dumps(sample, ensure_ascii=False)}"
    )

def ai_summary_openai(df: pd.DataFrame) -> List[str]:
    api_key = st.session_state.get("openai_key") or os.getenv("OPENAI_API_KEY")
    model = st.session_state.get("openai_model") or DEFAULTS["openai_model"]
    if not api_key:
        return summarize_locally(df)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_summary_prompt(df)}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        return summarize_locally(df) + [f"(Local fallback — OpenAI error: {e})"]
    lines = [l.strip("•- ").strip() for l in text.split("\n") if l.strip()]
    return lines[:5] if lines else [text]

def ai_summary_deepseek(df: pd.DataFrame) -> List[str]:
    api_key = st.session_state.get("deepseek_key") or os.getenv("DEEPSEEK_API_KEY")
    model = st.session_state.get("deepseek_model") or DEFAULTS["deepseek_model"]
    base = st.session_state.get("deepseek_base") or DEFAULTS["deepseek_base"]
    if not api_key:
        return summarize_locally(df)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_summary_prompt(df)}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        return summarize_locally(df) + [f"(Local fallback — DeepSeek error: {e})"]
    lines = [l.strip("•- ").strip() for l in text.split("\n") if l.strip()]
    return lines[:5] if lines else [text]

def ai_summary_anthropic(df: pd.DataFrame) -> List[str]:
    api_key = st.session_state.get("anthropic_key") or os.getenv("ANTHROPIC_API_KEY")
    model = st.session_state.get("anthropic_model") or DEFAULTS["anthropic_model"]
    if not api_key:
        return summarize_locally(df)
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": build_summary_prompt(df)}],
            temperature=0.2,
        )
        parts = []
        for b in resp.content:
            if isinstance(b, dict):
                parts.append(b.get("text", ""))
            else:
                parts.append(getattr(b, "text", ""))
        text = "".join(parts).strip() or "(empty)"
    except Exception as e:
        return summarize_locally(df) + [f"(Local fallback — Anthropic error: {e})"]
    lines = [l.strip("•- ").strip() for l in text.split("\n") if l.strip()]
    return lines[:5] if lines else [text]

def ai_summary_mistral(df: pd.DataFrame) -> List[str]:
    api_key = st.session_state.get("mistral_key") or os.getenv("MISTRAL_API_KEY")
    model = st.session_state.get("mistral_model") or DEFAULTS["mistral_model"]
    if not api_key:
        return summarize_locally(df)
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"model": model, "messages": [{"role": "user", "content": build_summary_prompt(df)}], "temperature": 0.2}
        r = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=body, timeout=20)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return summarize_locally(df) + [f"(Local fallback — Mistral error: {e})"]
    lines = [l.strip("•- ").strip() for l in text.split("\n") if l.strip()]
    return lines[:5] if lines else [text]

def ai_summary_ollama(df: pd.DataFrame) -> List[str]:
    base = st.session_state.get("ollama_base") or DEFAULTS["ollama_base"]
    model = st.session_state.get("ollama_model") or DEFAULTS["ollama_model"]
    try:
        url = f"{base.rstrip('/')}/api/chat"
        body = {"model": model, "messages": [{"role": "user", "content": build_summary_prompt(df)}]}
        r = requests.post(url, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        text = (data.get("message", {}) or {}).get("content", "") or data.get("response", "")
        text = str(text).strip() or "(empty)"
    except Exception as e:
        return summarize_locally(df) + [f"(Local fallback — Ollama error: {e})"]
    lines = [l.strip("•- ").strip() for l in text.split("\n") if l.strip()]
    return lines[:5] if lines else [text]

def summarize_with_ai(provider: str, df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["No tickets to summarize."]
    p = (provider or "").lower()
    if "openai" in p:
        return ai_summary_openai(df)
    if "anthropic" in p or "claude" in p:
        return ai_summary_anthropic(df)
    if "mistral" in p:
        return ai_summary_mistral(df)
    if "deepseek" in p:
        return ai_summary_deepseek(df)
    return ai_summary_ollama(df)  # default: Ollama

# -----------------------------
# About + quick tip
# -----------------------------
with st.expander("ℹ️ About this project"):
    st.markdown("""
**Vision.** A unified dashboard for modern Agile delivery.

**Tech.**
- 🖥️ *Frontend*: Streamlit (Python)
- 🤖 *AI*: OpenAI, Claude, Mistral, DeepSeek, Ollama
- ⚡ *Automation*: n8n (webhooks) — n8n creates real Jira issues from app submissions
- 📊 *Data*: Pandas + JSON

**Value.**
- Instant AI summaries
- 360° ticket view (filters, cards)
- Plug-and-play automations via n8n
""")

# -----------------------------
# Sidebar — settings (n8n & AI) + filters toggle
# -----------------------------
with st.sidebar:
    with st.expander("Settings (n8n & AI)", expanded=True):
        # n8n
        st.session_state.n8n_url = st.text_input("N8N Webhook URL", value=os.getenv("N8N_WEBHOOK_URL", ""))
        st.session_state.n8n_only = st.checkbox("Create via n8n only (no local save)", value=DEFAULT_N8N_ONLY)

        # Connectivity test
        if st.button("Test n8n connection"):
            url = st.session_state.get("n8n_url") or os.getenv("N8N_WEBHOOK_URL","")
            if not url:
                st.error("No N8N_WEBHOOK_URL set.")
            else:
                try:
                    r = requests.post(url, json={"action":"ping"}, timeout=8)
                    st.success(f"n8n reachable: HTTP {r.status_code}")
                    txt = r.text
                    st.code((txt[:500] + ("..." if len(txt) > 500 else "")) or "(empty)")
                except Exception as e:
                    st.error(f"n8n unreachable: {e}")

        st.write("---")

        # AI provider
        st.session_state.ai_provider = st.selectbox(
            "AI Provider",
            ["Ollama (local)", "OpenAI", "Anthropic (Claude)", "Mistral", "DeepSeek"],
            index=0
        )
        p = st.session_state.ai_provider
        if p == "OpenAI":
            st.session_state.openai_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY",""), type="password")
            st.session_state.openai_model = st.text_input("OPENAI_MODEL", value=DEFAULTS["openai_model"])
        elif p == "Anthropic (Claude)":
            st.session_state.anthropic_key = st.text_input("ANTHROPIC_API_KEY", value=os.getenv("ANTHROPIC_API_KEY",""), type="password")
            st.session_state.anthropic_model = st.text_input("ANTHROPIC_MODEL", value=DEFAULTS["anthropic_model"])
        elif p == "Mistral":
            st.session_state.mistral_key = st.text_input("MISTRAL_API_KEY", value=os.getenv("MISTRAL_API_KEY",""), type="password")
            st.session_state.mistral_model = st.text_input("MISTRAL_MODEL", value=DEFAULTS["mistral_model"])
        elif p == "DeepSeek":
            st.session_state.deepseek_key = st.text_input("DEEPSEEK_API_KEY", value=os.getenv("DEEPSEEK_API_KEY",""), type="password")
            st.session_state.deepseek_model = st.text_input("DEEPSEEK_MODEL", value=DEFAULTS["deepseek_model"])
            st.session_state.deepseek_base = st.text_input("DEEPSEEK_BASE_URL", value=DEFAULTS["deepseek_base"])

# -----------------------------
# Load tickets (demo JSON only)
# -----------------------------
tickets = load_demo_tickets()
st.session_state.tickets = tickets
df = to_df(tickets) if tickets else pd.DataFrame()

# -----------------------------
# Filters & view
# -----------------------------
with st.sidebar:
    st.write("---")
    st.header("Filters")
    q = st.text_input("Search (summary, description, key)")
    types = sorted(df["type"].dropna().unique().tolist()) if not df.empty else []
    statuses = sorted(df["status"].dropna().unique().tolist()) if not df.empty else []
    assignees = sorted([a for a in df["assignee"].dropna().unique().tolist() if a]) if not df.empty else []
    priorities = sorted(df["priority"].dropna().unique().tolist()) if not df.empty else []
    f_type = st.multiselect("Type", types, default=types)
    f_status = st.multiselect("Status", statuses, default=statuses)
    f_assignee = st.multiselect("Assignee", assignees)
    f_priority = st.multiselect("Priority", priorities)
    view = st.radio("View", ["Table", "Cards"], horizontal=True)

# Apply filters
fdf = df.copy()
if not fdf.empty:
    if f_type:     fdf = fdf[fdf["type"].isin(f_type)]
    if f_status:   fdf = fdf[fdf["status"].isin(f_status)]
    if f_assignee: fdf = fdf[fdf["assignee"].isin(f_assignee)]
    if f_priority: fdf = fdf[fdf["priority"].isin(f_priority)]
    if q:
        ql = q.lower()
        cols = ["summary", "description", "key"]
        mask = False
        for c in cols:
            if c in fdf.columns:
                mask = mask | fdf[c].astype(str).str.lower().str.contains(ql, na=False)
        fdf = fdf[mask]

# -----------------------------
# KPIs + alerts + chart
# -----------------------------
st.subheader("📊 Overview")
k1, k2, k3, k4, k5 = st.columns(5)

total = len(fdf)
done = int((fdf["status"]=="Done").sum()) if "status" in fdf else 0
wip = int((fdf["status"]=="In Progress").sum()) if "status" in fdf else 0
blocked = int((fdf["status"]=="Blocked").sum()) if "status" in fdf else 0

# Avg age (days)
try:
    ages = (pd.Timestamp.utcnow() - pd.to_datetime(fdf["created"])).dt.days
    age_avg = int(ages.mean()) if len(ages) else 0
except Exception:
    age_avg = 0

with k1:
    st.markdown(f"<div class='metric-card'><h3 style='margin:0;color:var(--brand-1);'>{total}</h3><p class='small-muted'>Tickets</p></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='metric-card'><h3 style='margin:0;color:var(--ok);'>{done}</h3><p class='small-muted'>✅ Done</p></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='metric-card'><h3 style='margin:0;color:var(--warn);'>{wip}</h3><p class='small-muted'>🔄 In Progress</p></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='metric-card'><h3 style='margin:0;color:var(--ko);'>{blocked}</h3><p class='small-muted'>🚫 Blocked</p></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='metric-card'><h3 style='margin:0;color:var(--brand-2);'>{age_avg} d</h3><p class='small-muted'>Avg age</p></div>", unsafe_allow_html=True)

if wip > WIP_LIMIT:
    st.warning(f"High WIP ({wip}) — consider limiting 'In Progress' (limit {WIP_LIMIT}).")

# Simple chart
if not fdf.empty and "status" in fdf.columns:
    st.subheader("📈 Status distribution")
    st.bar_chart(fdf["status"].value_counts())

# -----------------------------
# Exports + AI Summary
# -----------------------------
exp1, exp2 = st.columns(2)
with exp1:
    if not fdf.empty:
        st.download_button("⬇️ Export CSV (filtered)",
            data=fdf.to_csv(index=False).encode("utf-8"),
            file_name="tickets_filtered.csv", mime="text/csv")
with exp2:
    if not fdf.empty:
        st.download_button("⬇️ Export JSON (filtered)",
            data=json.dumps(fdf.to_dict(orient="records"), ensure_ascii=False, indent=2),
            file_name="tickets_filtered.json", mime="application/json")

if st.button("🧠 AI summary of filtered tickets"):
    if fdf.empty:
        st.info("No tickets to summarize.")
    else:
        with st.spinner("Generating summary..."):
            bullets = summarize_with_ai(st.session_state.get("ai_provider", "Ollama (local)"), fdf)
        st.subheader("Summary")
        for b in bullets[:5]:
            st.markdown(f"- {b}")

# -----------------------------
# Tickets view (table/cards + pagination)
# -----------------------------
st.subheader("Tickets")
if fdf.empty:
    st.info("No tickets yet. Generate the demo dataset or create a ticket below.")
else:
    if view == "Table":
        # Put useful columns first
        preferred = ["key","type","status","priority","assignee","summary","created","updated","due_date"]
        cols = [c for c in preferred if c in fdf.columns] + [c for c in fdf.columns if c not in preferred]
        st.dataframe(fdf[cols], use_container_width=True, height=520)
    else:
        PAGE_SIZE = st.slider("Cards per page", 6, 30, 9, 3)
        total_pages = (len(fdf) + PAGE_SIZE - 1) // PAGE_SIZE
        page = st.number_input("Page", 1, max(1, total_pages), 1) - 1
        slice_df = fdf.iloc[page*PAGE_SIZE:(page+1)*PAGE_SIZE]

        n = len(slice_df)
        cards_per_row = 3
        rows = (n + cards_per_row - 1) // cards_per_row
        for i in range(rows):
            cols = st.columns(cards_per_row)
            for j, col in enumerate(cols):
                idx = i * cards_per_row + j
                if idx >= n: break
                row = slice_df.iloc[idx]
                with col:
                    st.markdown(f"### {row.get('key','?')} — {row.get('summary','')}")
                    st.caption(f"{row.get('type','')} • {row.get('status','')} • {row.get('priority','')}")
                    st.write(f"**Assignee**: {row.get('assignee','—')}")
                    st.write(f"**Labels**: {row.get('labels','')}")
                    st.write(f"**Components**: {row.get('components','')}")
                    if row.get("due_date"):
                        st.write(f"**Due**: {row.get('due_date')}")

st.divider()

# Create Ticket form (sends to n8n; optionally skips local save)

st.markdown("""
<div style='background:#e8f4fd; padding:1rem; border-radius:10px; margin:1rem 0;'>
  <h2 style='color:var(--brand-1); margin:0;'>➕ Create a new ticket</h2>
  <p style='margin:.4rem 0 0 0; color:#555;'>This app POSTs to your n8n webhook. n8n should create the real Jira issue.</p>
</div>
""", unsafe_allow_html=True)



with st.form("create_ticket"):
    # ✅ ADD : Write the Jira key (ex: TES)
    project_key = st.text_input("Project key (Jira)", value="TES")

    c1, c2, c3 = st.columns(3)
    issue_type = c1.selectbox("Type", ["Task", "Story", "Bug", "Epic"])
    status = c2.selectbox("Status", ["Backlog", "To Do", "In Progress", "In Review", "Blocked", "Done"], index=1)
    priority = c3.selectbox("Priority", ["Lowest", "Low", "Medium", "High", "Highest"], index=2)

    summary = st.text_input("Summary", placeholder="e.g., Add dark mode to analytics")
    description = st.text_area("Description", placeholder="Context / Problem / Solution...")

    c4, c5, c6 = st.columns(3)
    assignee = c4.text_input("Assignee (optional)")
    sprint = c5.text_input("Sprint (optional)", placeholder="Sprint 23")
    story_points = c6.number_input("Story points (optional)", min_value=0, step=1, value=0)

    labels = st.text_input("Labels (comma-separated)", value="demo")
    components = st.text_input("Components (comma-separated)", value="API")

    # Links (demo-only)
    epics = [t["key"] for t in tickets if t.get("type") == "Epic"] if tickets else []
    parents = [t["key"] for t in tickets if t.get("type") in ("Task", "Story")] if tickets else []
    epic_link = st.selectbox("Epic link (optional, demo)", [""] + epics)
    parent = st.selectbox("Parent (optional, for Sub-task; demo)", [""] + parents)

    add_due = st.checkbox("Add a due date?", value=False)
    if add_due:
        due_date = st.date_input("Due date")
    else:
        due_date = None

    submitted = st.form_submit_button("Create ticket")


# Submission - Out of the context with st.form()
if submitted:
    if not summary or len(summary.strip()) < 5:
        st.warning("Summary is required (≥ 5 characters).")
    else:
        now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        project = tickets[0]["project"] if tickets else "DEMO"
        new_key = next_key(tickets, project=project)
        due_iso = (
            datetime.combine(due_date, time(0, 0)).strftime("%Y-%m-%dT%H:%M:%SZ")
            if add_due and due_date else None
        )
        
        new_ticket = {
            "id": str(uuid.uuid4()),
            "key": new_key,
            "project": project_key,
            "type": issue_type,
            "summary": summary,
            "description": description or "—",
            "status": status,
            "priority": priority,
            "assignee": assignee or None,
            "reporter": "Streamlit Demo",
            "created": now_iso,
            "updated": now_iso,
            "due_date": due_iso,
            "labels": [s.strip() for s in labels.split(",") if s.strip()] if labels else [],
            "components": [s.strip() for s in components.split(",") if s.strip()] if components else [],
            "sprint": sprint or None,
            "story_points": int(story_points) if story_points else None,
            "epic_link": epic_link or None,
            "parent": parent or None,
            "attachments": [],
            "watchers": 0,
            "url": f"https://jira.example.com/browse/{new_key}",
            "projectKey": project_key,
        }

        # Send to n8n
        status_code, body = post_to_n8n({"action": "create_ticket", "ticket": new_ticket})

        # Manage depending on the mode
        if st.session_state.get("n8n_only"):
            if status_code == "no_url":
                st.warning("n8n-only is ON but N8N_WEBHOOK_URL is not set.")
            elif str(status_code).startswith("error"):
                st.error(f"n8n webhook error: {status_code}")
            else:
                try:
                    if isinstance(body, dict):
                        jira_key = body.get("jiraKey") or body.get("key")
                        jira_url = body.get("url")
                        if jira_key and jira_url:
                            st.success("✅ Ticket successfully created in Jira!")
                            
                            st.subheader("Ticket details created")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.info(f"**Jira key:** {jira_key}")
                                st.info(f"**Project:** {project_key}")
                                st.info(f"**Type:** {issue_type}")
                                st.info(f"**Summary:** {summary}")
                            
                            with col2:
                                st.info(f"**Status:** {status}")
                                st.info(f"**Priority:** {priority}")
                                if assignee:
                                    st.info(f"**Assigned to:** {assignee}")
                                st.info(f"**URL:** [Open in Jira]({jira_url})")
                        else:
                            st.success(f"Ticket sent to n8n (HTTP {status_code}).")
                    else:
                        st.success(f"Ticket sent to n8n (HTTP {status_code}).")
                except Exception as e:
                    st.success(f"Ticket sent to n8n (HTTP {status_code}).")
        else:
            # Local mode + n8n
            tickets.append(new_ticket)
            st.session_state.tickets = tickets
            save_tickets(tickets)
            
            st.success("✅ Ticket successfully created!")
            st.subheader("Created ticket details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Key:** {new_key}")
                st.info(f"**Project:** {project_key}")
                st.info(f"**Type:** {issue_type}")
                st.info(f"**Summary:** {summary}")
            
            with col2:
                st.info(f"**Status:** {status}")
                st.info(f"**Priority:** {priority}")
                if assignee:
                    st.info(f"**Assigned to:** {assignee}")
                st.info(f"**URL:** [Lien local]({new_ticket['url']})")
            
            if status_code == "no_url":
                st.info("Ticket created locally. (n8n webhook not configured)")
            elif str(status_code).startswith("error"):
                st.warning(f"Ticket created locally. Webhook error: {status_code}")
            else:
                msg = f"Ticket created locally and sent to n8n (HTTP {status_code})."
                try:
                    if isinstance(body, dict):
                        jira_key = body.get("jiraKey") or body.get("key")
                        if jira_key:
                            msg = f"Ticket created locally; n8n created Jira issue: {jira_key}"
                            st.info(f"**Jira Ticket created:** {jira_key}")
                except Exception:
                    pass
                st.success(msg)
# -----------------------------
# Signature / CTA
# -----------------------------
st.markdown("---")
c1, c2, c3 = st.columns([1,2,1])
with c2:
    st.markdown("""
    <div style='text-align:center; padding:1rem; background:rgba(127,127,127,.07); border-radius:10px;'>
      <h4 style='margin:0 0 .3rem 0;'>📫 Built by <strong>Yassin Zehar</strong></h4>
            <p style='margin:.2rem 0;'> Let's connect 🔗 <a href='https://linkedin.com/in/yassin-zehar/en' target='_blank'>LinkedIn</a> • 
         💻 <a href='https://github.com/YassinAnalytics' target='_blank'>GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

st.caption("Set N8N_WEBHOOK_URL in your environment (or in the sidebar). Toggle 'n8n only' to avoid local JSON writes and let n8n/Jira be the source of truth.")
