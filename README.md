#Streamlit-Jira-Control-Center 
Automate Jira Ticket Creation from a Streamlit App (via n8n)

A portfolio-friendly demo showing how a Streamlit app can create Jira issues through an n8n webhook. Includes a rich dashboard, optional AI summaries, and a safe “demo-only” mode that works without any external services.

##🔎 What you get

- Streamlit dashboard with table/cards views, filters, KPIs, and exports

- Create Ticket form that POSTs to your n8n webhook

- Two modes

- Demo (no external calls): generates/loads sample data locally

- Live (production): posts to n8n → Jira REST API

- Optional AI summaries (OpenAI/Claude/Mistral/DeepSeek/Ollama) with local fallback

- De-dup & validation in n8n to avoid empty/duplicate tickets

🚀 ##Quick start
- Option A — Portfolio demo (no external services)

Create a virtual env and install deps:
pip install -r requirements.txt

Run:
python -m streamlit run app.py

In the app:

If you don’t see demo data yet, use the “Generate demo dataset (60 tickets)” button.

Use Filters, AI summary (optional), and Create a new ticket form.

Leave “Create via n8n only” OFF (default in this repo) so tickets are shown locally.

- Option B — Live Jira creation (n8n + Jira)

Import & activate the n8n workflow

Import the provided n8n JSON (the template is designed to be reused).

Set the Webhook node Response mode = lastNode.

Activate the workflow and copy the Production URL (not the Test URL).

Set Streamlit secret N8N_WEBHOOK_URL

Local: create a .env file next to app.py:


N8N_WEBHOOK_URL=https://YOUR-N8N-HOST/webhook/your-path
N8N_ONLY=true

Streamlit Community Cloud: go to Settings → Secrets and add:


N8N_WEBHOOK_URL="https://YOUR-N8N-HOST/webhook/your-path"
N8N_ONLY="true"

Jira credentials in n8n

In the HTTP Request node (or Jira credential), use Jira Software Cloud API auth (email + API token).

Endpoint: https://<your-domain>.atlassian.net/rest/api/3/issue

In the app sidebar, click “Test n8n connection” (optional).

Submit the Create a new ticket form. You should get a Jira key back.

| Key                 | What it does                                       | Required?              | Example                                  |
| ------------------- | -------------------------------------------------- | ---------------------- | ---------------------------------------- |
| `N8N_WEBHOOK_URL`   | n8n production webhook URL                         | No (demo) / Yes (live) | `https://n8n.example.com/webhook/abc123` |
| `N8N_ONLY`          | If `true`, skip local save; n8n is source of truth | No                     | `true` or `false`                        |
| `OPENAI_API_KEY`    | For OpenAI summaries                               | Optional               | `sk-...`                                 |
| `ANTHROPIC_API_KEY` | For Claude summaries                               | Optional               | `...`                                    |
| `MISTRAL_API_KEY`   | For Mistral summaries                              | Optional               | `...`                                    |
| `DEEPSEEK_API_KEY`  | For DeepSeek summaries                             | Optional               | `...`                                    |
| `DEEPSEEK_BASE_URL` | DeepSeek base URL                                  | Optional               | `https://api.deepseek.com/v1`            |


🧩 ##How it works

- Streamlit form collects ticket details (project key, type, summary, description, priority, story points, etc.).

- POST to n8n (if N8N_WEBHOOK_URL is set or the user inputs it in the sidebar).

- n8n workflow

- Anti-double: blocks empty/invalid requests and prevents duplicates (simple in-memory check).

Process data: builds a valid Jira payload:

{
  "fields": {
    "project": { "key": "TES" },
    "summary": "Implement dark mode",
    "issuetype": { "name": "Task" },
    "description": { "type": "doc", "version": 1, "content": [ ... ] },
    "priority": { "name": "Medium" },
    "customfield_10016": 5    // Story Points (ID may differ on your Jira)
  }
}


HTTP Request → Jira: POST /rest/api/3/issue

Return to app: { ok: true, jiraKey: "TES-123", url: "https://<your>.atlassian.net/browse/TES-123" }

Streamlit UI displays the created key + link (or local confirmation in demo mode).


🧪 ##Test the webhook (without the app)


curl -X POST "$N8N_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "create_ticket",
    "ticket": {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "projectKey": "TES",
      "type": "Task",
      "summary": "Webhook test from cURL",
      "description": "Created from a direct webhook call.",
      "priority": "Medium",
      "story_points": 3
    }
  }'

  
Expected (from n8n):
{ "ok": true, "jiraKey": "TES-123", "url": "https://<your>.atlassian.net/browse/TES-123" }

🧠 ##AI summaries (optional)

Choose provider from the sidebar (OpenAI, Claude, Mistral, DeepSeek, Ollama).

If no API key is set, the app uses a local fallback summarizer (no external calls).


🔐 ##Security notes

- Never commit secrets (.env, API keys, webhook URLs).

- Use Streamlit Secrets (cloud) or a local .env.

- In n8n, keep your webhook path private and use the Production URL in the app.

- Jira assignee in Cloud usually requires accountId. The workflow intentionally avoids setting assignee.name.

🧯 ##Troubleshooting

- “400 - Invalid request payload” from Jira

- Ensure the n8n HTTP node sends raw JSON body (not [object Object]).

- Headers: Content-Type: application/json, Accept: application/json.

- Payload must be { "fields": { ... } }, not just fields directly.

- Story Points field ID may differ (often customfield_10016). Adjust if needed.

- Duplicate empty tickets

- Don’t run the HTTP Request node manually in n8n. Trigger only via Webhook.

- Keep the anti-double code and the IF branching as in the template.

- In Streamlit, the form submits once; avoid triggering multiple payloads on reruns.

- No data appears in the dashboard

Add tickets_jira_demo.json or click “Generate demo dataset (60 tickets)” in the app.


🧪 ##Requirements

- Minimal (for demo + optional OpenAI fallback):
- streamlit>=1.31
- pandas>=2.1
- requests>=2.31
- python-dotenv>=1.0
- openai>=1.43 ; python_version >= "3.8"

If you won’t use OpenAI, you can drop it from requirements.txt. The app will still run and use local summaries.


📦 ##Deploy to Streamlit Community Cloud

- Push app.py and requirements.txt to a GitHub repo.

- In Streamlit Cloud, New app → select your repo/branch.

(Optional) Add Secrets:
N8N_WEBHOOK_URL="https://YOUR-N8N/webhook/your-path"
N8N_ONLY="true"
OPENAI_API_KEY="..."

Deploy. Your app will be live at something like:

https://<your-repo-name>-<your-username>.streamlit.app


🙋 ##FAQ

Can I publish without any secrets?
Yes. It will run as a portfolio demo with local data and no external calls.

Do I need the demo JSON?
No. If it’s missing, the app can generate 60 realistic tickets for you.

Where do I put my n8n URL?
In .env (local) or Streamlit Secrets (cloud) as N8N_WEBHOOK_URL.

Will this overwrite anything in Jira?
No. It only creates issues via the Jira REST API when you enable the live mode and submit the form.
