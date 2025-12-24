from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, create_engine


class Student(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: str = Field(index=True, unique=True)  # e.g. "ESILV2025-001"
    first_name: str
    last_name: str
    email: str = Field(index=True, unique=True)
    program: str = Field(default="")
    year: int = Field(default=0)

    created_at: datetime = Field(default_factory=datetime.utcnow)


def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


def init_db(engine) -> None:
    SQLModel.metadata.create_all(engine)