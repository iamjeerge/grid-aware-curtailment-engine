"""Database configuration and models for PostgreSQL.

This module provides SQLAlchemy models and database connection
for persistent storage of optimization results.
"""

from __future__ import annotations

import os
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Database URL from environment or default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/curtailment_engine",
)

# Create engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class OptimizationModel(Base):
    """SQLAlchemy model for optimization results."""

    __tablename__ = "optimizations"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(200), nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Configuration (stored as JSON)
    scenario_config = Column(JSON, nullable=False)
    battery_config = Column(JSON, nullable=False)
    strategies = Column(JSON, nullable=False)

    # Results (stored as JSON)
    results = Column(JSON, nullable=True)
    comparison = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)


class DemoScenarioModel(Base):
    """SQLAlchemy model for demo scenarios."""

    __tablename__ = "demo_scenarios"

    id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    scenario_type = Column(String(50), nullable=False)
    config = Column(JSON, nullable=False)
    battery = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def drop_db() -> None:
    """Drop all database tables."""
    Base.metadata.drop_all(bind=engine)
