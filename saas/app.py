"""
Auto Job SaaS — web app.

    .venv\\Scripts\\python -m uvicorn saas.app:app --port 8000

Users sign up, upload a resume (which becomes their job-apply skill), connect Naukri,
and switch on auto-apply. The scheduler then refreshes their Naukri profile once a day
and applies to matching jobs up to their daily limit.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, EmailStr, Field  # noqa: E402
from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from saas import runs  # noqa: E402
from saas.config import COOKIE_SECURE, HARD_DAILY_LIMIT, SCHEDULER_ENABLED, SESSION_DAYS  # noqa: E402
from saas.db import (Application, NaukriAccount, Question, Resume, Run, Settings, User,  # noqa: E402
                     applied_today, get_db, get_or_create, init_db)
from saas.resume import process_resume_async, save_upload  # noqa: E402
from saas.security import (COOKIE_NAME, current_user, encrypt, hash_password, make_token,  # noqa: E402
                           verify_password)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "urllib3", "google", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    runs.recover_stale_runs()
    if SCHEDULER_ENABLED:
        runs.start_scheduler()
    yield


app = FastAPI(title="Auto Job", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class SignupIn(BaseModel):
    name: str = Field("", max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


def _set_cookie(resp: Response, user_id: int) -> None:
    resp.set_cookie(COOKIE_NAME, make_token(user_id), httponly=True, samesite="lax", secure=COOKIE_SECURE,
                    max_age=SESSION_DAYS * 86400)


@app.post("/api/auth/signup")
def signup(body: SignupIn, resp: Response, db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "An account with this email already exists.")
    user = User(email=email, name=body.name.strip(), password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    _set_cookie(resp, user.id)
    return {"ok": True}


@app.post("/api/auth/login")
def login(body: LoginIn, resp: Response, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Wrong email or password.")
    _set_cookie(resp, user.id)
    return {"ok": True}


@app.post("/api/auth/logout")
def logout(resp: Response):
    resp.delete_cookie(COOKIE_NAME)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Dashboard state
# ---------------------------------------------------------------------------

def _run_out(r: Optional[Run]) -> Optional[dict]:
    if not r:
        return None
    return {"id": r.id, "kind": r.kind, "trigger": r.trigger, "status": r.status, "summary": r.summary,
            "applied": r.applied, "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None}


@app.get("/api/me")
def me(user: User = Depends(current_user), db: Session = Depends(get_db)):
    res = db.scalar(select(Resume).where(Resume.user_id == user.id))
    acc = get_or_create(db, NaukriAccount, user.id)
    st = get_or_create(db, Settings, user.id)
    db.commit()
    counts = dict(db.execute(select(Application.status, func.count(Application.id))
                             .where(Application.user_id == user.id).group_by(Application.status)).all())
    last = db.scalar(select(Run).where(Run.user_id == user.id).order_by(Run.id.desc()).limit(1))
    active = runs.active_run(user.id)
    open_q = db.scalar(select(func.count(Question.id)).where(Question.user_id == user.id, Question.answer == ""))
    return {
        "user": {"name": user.name, "email": user.email},
        "resume": None if not res else {
            "filename": res.filename, "status": res.status, "error": res.error,
            "uploaded_at": res.uploaded_at.isoformat(), "skill": res.skill,
            "profile": {k: v for k, v in res.profile.items() if k in (
                "name", "email", "phone", "location", "education", "experience_years", "summary")},
        },
        "naukri": {"email": acc.email, "status": acc.status, "message": acc.status_message,
                   "has_password": bool(acc.password_enc),
                   "last_login_at": acc.last_login_at.isoformat() if acc.last_login_at else None,
                   "last_profile_refresh": acc.last_profile_refresh.isoformat() if acc.last_profile_refresh else None,
                   "last_profile_refresh_note": acc.last_profile_refresh_note},
        "settings": {c: getattr(st, c) for c in ("auto_apply", "daily_profile_refresh", "daily_limit", "min_match",
                                                 "keywords", "locations", "run_hour")},
        "stats": {"applied_today": applied_today(db, user.id), "by_status": counts, "open_questions": open_q or 0},
        "active_run": None if not active else {"kind": active.kind, "run_id": active.run_id,
                                               "applied": active.applied, "log": active.lines[-12:]},
        "last_run": _run_out(last),
        "hard_daily_limit": HARD_DAILY_LIMIT,
    }


# ---------------------------------------------------------------------------
# Resume + skill
# ---------------------------------------------------------------------------

@app.post("/api/resume")
async def upload_resume(file: UploadFile = File(...), user: User = Depends(current_user),
                        db: Session = Depends(get_db)):
    data = await file.read()
    try:
        path = save_upload(user.id, file.filename or "resume.pdf", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    res = db.scalar(select(Resume).where(Resume.user_id == user.id)) or Resume(user_id=user.id)
    res.filename, res.path, res.status, res.error = path.name, str(path), "processing", ""
    res.uploaded_at = datetime.utcnow()
    db.add(res)
    db.commit()
    process_resume_async(user.id)
    return {"ok": True}


class SkillIn(BaseModel):
    skill: dict


@app.put("/api/skill")
def update_skill(body: SkillIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    res = db.scalar(select(Resume).where(Resume.user_id == user.id))
    if not res or res.status != "ready":
        raise HTTPException(400, "Upload a resume first.")
    merged = {**res.skill, **body.skill}
    res.skill_json = json.dumps(merged, ensure_ascii=False)
    db.commit()
    return {"ok": True, "skill": merged}


@app.post("/api/skill/rebuild")
def rebuild_skill(user: User = Depends(current_user), db: Session = Depends(get_db)):
    res = db.scalar(select(Resume).where(Resume.user_id == user.id))
    if not res:
        raise HTTPException(400, "Upload a resume first.")
    res.status = "processing"
    db.commit()
    process_resume_async(user.id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Naukri account
# ---------------------------------------------------------------------------

class NaukriIn(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field("", max_length=255)


@app.post("/api/naukri/connect")
def naukri_connect(body: NaukriIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    acc = get_or_create(db, NaukriAccount, user.id)
    acc.email = body.email.strip()
    if body.password:
        acc.password_enc = encrypt(body.password)
    acc.status, acc.status_message = "not_connected", "Checking sign-in..."
    db.commit()
    ok, msg = runs.start(user.id, "login")
    if not ok:
        raise HTTPException(409, msg)
    return {"ok": True}


@app.post("/api/naukri/check")
def naukri_check(user: User = Depends(current_user)):
    """Re-check the saved Naukri session (and stored password, if any) without changing anything."""
    ok, msg = runs.start(user.id, "login")
    if not ok:
        raise HTTPException(409, msg)
    return {"ok": True}


@app.post("/api/naukri/browser-login")
def naukri_browser_login(user: User = Depends(current_user)):
    """Opens a visible browser on the server machine (local installs) for OTP / CAPTCHA sign-in."""
    ok, msg = runs.start(user.id, "manual_login")
    if not ok:
        raise HTTPException(409, msg)
    return {"ok": True}


@app.delete("/api/naukri")
def naukri_disconnect(user: User = Depends(current_user), db: Session = Depends(get_db)):
    acc = get_or_create(db, NaukriAccount, user.id)
    acc.email = acc.password_enc = ""
    acc.status, acc.status_message = "not_connected", "Disconnected"
    st = get_or_create(db, Settings, user.id)
    st.auto_apply = False
    db.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Settings, runs, applications, questions
# ---------------------------------------------------------------------------

class SettingsIn(BaseModel):
    auto_apply: Optional[bool] = None
    daily_profile_refresh: Optional[bool] = None
    daily_limit: Optional[int] = Field(None, ge=1, le=HARD_DAILY_LIMIT)
    min_match: Optional[int] = Field(None, ge=0, le=100)
    keywords: Optional[str] = Field(None, max_length=500)
    locations: Optional[str] = Field(None, max_length=500)
    run_hour: Optional[int] = Field(None, ge=0, le=23)


@app.put("/api/settings")
def update_settings(body: SettingsIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    st = get_or_create(db, Settings, user.id)
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(st, k, v)
    db.commit()
    return {"ok": True}


class RunIn(BaseModel):
    kind: str = Field(pattern="^(apply|dry_run|profile_refresh)$")


@app.post("/api/runs")
def start_run(body: RunIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    acc = get_or_create(db, NaukriAccount, user.id)
    if acc.status != "connected":
        raise HTTPException(400, "Connect your Naukri account first.")
    ok, msg = runs.start(user.id, body.kind)
    if not ok:
        raise HTTPException(409, msg)
    return {"ok": True}


@app.post("/api/runs/stop")
def stop_run(user: User = Depends(current_user)):
    return {"ok": runs.stop(user.id)}


@app.get("/api/runs")
def list_runs(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(Run).where(Run.user_id == user.id).order_by(Run.id.desc()).limit(20)).all()
    return [_run_out(r) for r in rows]


@app.get("/api/applications")
def list_applications(status: str = "", limit: int = 100, user: User = Depends(current_user),
                      db: Session = Depends(get_db)):
    q = select(Application).where(Application.user_id == user.id)
    if status:
        q = q.where(Application.status == status)
    rows = db.scalars(q.order_by(Application.id.desc()).limit(min(limit, 500))).all()
    return [{"id": a.id, "title": a.title, "company": a.company, "url": a.url, "score": a.match_score,
             "status": a.status, "message": a.message, "created_at": a.created_at.isoformat()} for a in rows]


@app.get("/api/questions")
def list_questions(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(Question).where(Question.user_id == user.id)
                      .order_by((Question.answer == "").desc(), Question.id.desc()).limit(200)).all()
    return [{"id": q.id, "question": q.question, "options": q.options, "answer": q.answer} for q in rows]


class AnswerIn(BaseModel):
    answer: str = Field(max_length=1000)


@app.put("/api/questions/{qid}")
def answer_question(qid: int, body: AnswerIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    q = db.get(Question, qid)
    if not q or q.user_id != user.id:
        raise HTTPException(404, "Question not found")
    q.answer = body.answer.strip()
    db.commit()
    return {"ok": True}


class NewQuestionIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    answer: str = Field(min_length=1, max_length=1000)


@app.post("/api/questions")
def add_question(body: NewQuestionIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.add(Question(user_id=user.id, question=body.question.strip(), answer=body.answer.strip()))
    db.commit()
    return {"ok": True}
