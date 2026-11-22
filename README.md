# Auto Job — standalone job-apply backend

Veronica's job agents (Naukri / Indeed / LinkedIn quick-apply + company career
portals) as a terminal app. No UI, no voice. Veronica still has its own copy;
this folder is self-contained and can be moved anywhere.

## Setup (Windows)

```bash
cd auto_job
uv venv --python 3.12
uv pip install -r requirements.txt
```

- Python 3.12 or 3.13 (not 3.14). `.python-version` makes `uv venv` choose 3.12.
- Uses your installed **Microsoft Edge**. If Edge is missing: `.venv\Scripts\playwright install msedge`.
- `.env` holds the LLM keys (NVIDIA / Groq / Gemini). See `.env.example`.

## Run

```bash
.venv\Scripts\python main.py              # interactive shell, type: help
.venv\Scripts\python main.py board naukri # continuous quick-apply (naukri | indeed | linkedin)
.venv\Scripts\python main.py board linkedin -k "AI Engineer, ML Engineer" -l Bengaluru -n 20
.venv\Scripts\python main.py board indeed --dry-run   # fills everything, never submits
.venv\Scripts\python main.py url https://company.com/careers/job/123
.venv\Scripts\python main.py auto "AI Engineer"       # Naukri search, then company-site jobs
.venv\Scripts\python main.py saved                    # apply to saved company-site jobs
.venv\Scripts\python main.py report
.venv\Scripts\python main.py teach "Notice period: 30 days" --answer
```

Ctrl+C stops after the current step.

## You stay in the loop

- **Board run:** shows the plan; one yes covers the whole run. If it pauses
  (sign-in, CAPTCHA, daily limit), fix it in the Edge window and press Enter.
- **Company sites:** every account creation and every final Submit asks `[y/n]`.
- **Screening questions** your CV/profile don't answer are asked in the
  terminal while it runs. Your answer is saved and reused next time.
- Portal passwords go to Windows Credential Manager (`Veronica-JobPortals`).

## Where things live

| Path | What |
|---|---|
| `data/profile/ABOUT_ME.md`, `data/profile/cv/` | your profile and CV (copied from Veronica) |
| `data/job_applications/` | candidate profile, application log, saved company-site queue |
| `data/job_agent/memories/` | what the agent learned per site, saved answers |
| `data/autojob.log` | full debug log |
| `secrets/token.json` | optional Gmail token for reading verification codes |

The bots use the Edge profile `%LOCALAPPDATA%\VeronicaEdgeProfile`, the same
one Veronica uses, so your board logins carry over. Don't run both apps at
once. Set `AUTOJOB_EDGE_PROFILE` in `.env` to use a separate profile.

Data is copied, not shared. Applications made here don't show up in Veronica's
log, and the other way round.

## SaaS web app (multi-user)

```bash
uv pip install -r requirements.txt
.venv\Scripts\python -m uvicorn saas.app:app --port 8000
```

Open http://localhost:8000, create an account, then:

1. **Resume & skill** – upload a PDF/DOCX. It is parsed and turned into an *apply skill*:
   search keywords, target titles, core/secondary skills, title words to avoid, experience,
   standard answers. Every field is editable.
2. **Naukri** – save the Naukri login (password encrypted at rest). If Naukri wants an OTP or
   CAPTCHA, use *Sign in with a browser window* once; the session is kept in the user's own
   browser profile (`data/saas/users/<id>/`).
3. **Overview** – turn on **Auto-apply**. Every day from the start hour the scheduler:
   - makes a small edit to the Naukri profile once (toggles a trailing full stop on the resume
     headline, or re-uploads the resume) so it shows as updated today,
   - searches Naukri with the skill's keywords, scores each job against the resume, and applies
     to jobs at or above the minimum match, up to the daily limit (max 50).
4. **Questions** – screening questions the resume can't answer are skipped and listed here;
   answer once and later runs use the answer.

| Path | What |
|---|---|
| `saas/app.py` | FastAPI app + JSON API |
| `saas/naukri_worker.py` | per-user Naukri login, daily profile refresh, apply loop |
| `saas/runs.py` | run manager + daily scheduler |
| `saas/resume.py` | resume -> apply skill |
| `saas/static/` | dashboard (HTML/JS/CSS) |
| `data/saas/` | database, keys, per-user resumes and browser profiles |
