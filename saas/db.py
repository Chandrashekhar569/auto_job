"""
Database models. SQLite by default (data/saas/autojob.db); set SAAS_DATABASE_URL
to a Postgres URL for production.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Iterator, Optional

from sqlalchemy import (Boolean, Date, DateTime, ForeignKey, Integer, String, Text, create_engine,
                        func, select)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from saas.config import DATABASE_URL, DEFAULT_DAILY_LIMIT

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Resume(Base):
    """The user's current resume and the job-apply 'skill' built from it."""
    __tablename__ = "resumes"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(String(1024))
    raw_text: Mapped[str] = mapped_column(Text, default="")
    profile_json: Mapped[str] = mapped_column(Text, default="{}")   # parsed CV fields
    skill_json: Mapped[str] = mapped_column(Text, default="{}")     # the apply skill (keywords, rules, answers)
    status: Mapped[str] = mapped_column(String(32), default="processing")  # processing | ready | error
    error: Mapped[str] = mapped_column(Text, default="")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    @property
    def profile(self) -> dict:
        return _loads(self.profile_json)

    @property
    def skill(self) -> dict:
        return _loads(self.skill_json)


class NaukriAccount(Base):
    __tablename__ = "naukri_accounts"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), default="")
    password_enc: Mapped[str] = mapped_column(Text, default="")
    # not_connected | connected | needs_attention | error
    status: Mapped[str] = mapped_column(String(32), default="not_connected")
    status_message: Mapped[str] = mapped_column(Text, default="")
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_profile_refresh: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_profile_refresh_note: Mapped[str] = mapped_column(Text, default="")


class Settings(Base):
    __tablename__ = "settings"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    auto_apply: Mapped[bool] = mapped_column(Boolean, default=False)
    daily_profile_refresh: Mapped[bool] = mapped_column(Boolean, default=True)
    daily_limit: Mapped[int] = mapped_column(Integer, default=DEFAULT_DAILY_LIMIT)
    min_match: Mapped[int] = mapped_column(Integer, default=60)
    keywords: Mapped[str] = mapped_column(Text, default="")     # comma separated; empty = from the skill
    locations: Mapped[str] = mapped_column(Text, default="")    # comma separated; empty = main cities + all India
    run_hour: Mapped[int] = mapped_column(Integer, default=9)   # local hour the daily run may start from


class Application(Base):
    __tablename__ = "applications"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("runs.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(512), default="")
    company: Mapped[str] = mapped_column(String(512), default="")
    url: Mapped[str] = mapped_column(String(2048), default="", index=True)
    match_score: Mapped[int] = mapped_column(Integer, default=0)
    # applied | skipped | external | needs_user | error | dry_run | unconfirmed
    status: Mapped[str] = mapped_column(String(32), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))     # apply | profile_refresh | login
    trigger: Mapped[str] = mapped_column(String(32), default="manual")   # manual | scheduler
    status: Mapped[str] = mapped_column(String(32), default="queued")    # queued | running | done | error | stopped
    summary: Mapped[str] = mapped_column(Text, default="")
    applied: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Question(Base):
    """A screening question the resume did not answer. The user answers it once; later runs reuse it."""
    __tablename__ = "questions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    options_json: Mapped[str] = mapped_column(Text, default="[]")
    answer: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    @property
    def options(self) -> list:
        v = _loads(self.options_json)
        return v if isinstance(v, list) else []


def _loads(text: str) -> Any:
    try:
        return json.loads(text or "{}")
    except Exception:
        return {}


def init_db() -> None:
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def get_or_create(s: Session, model, user_id: int):
    obj = s.scalar(select(model).where(model.user_id == user_id))
    if obj is None:
        obj = model(user_id=user_id)
        s.add(obj)
        s.flush()
    return obj


def applied_today(s: Session, user_id: int) -> int:
    start = datetime.combine(date.today(), datetime.min.time())
    return s.scalar(select(func.count(Application.id)).where(
        Application.user_id == user_id, Application.status == "applied",
        Application.created_at >= start)) or 0
