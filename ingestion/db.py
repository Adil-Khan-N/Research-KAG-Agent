"""
Database connection and session management.
Single place to get a Postgres connection — import this everywhere.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql://kag:kag@localhost:5432/kagdb")

def _resolve_database_url(url: str) -> str:
    """
    Ensure the PostgreSQL URL specifies a working driver (psycopg2 or psycopg 3).
    Avoids 'ImportError: no pq wrapper available' when psycopg 3 is installed
    without binary wheels.
    """
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        prefix = "postgresql://" if url.startswith("postgresql://") else "postgres://"
        # Check if psycopg (v3) has a working libpq wrapper
        has_psycopg3 = False
        try:
            import psycopg
            from psycopg import pq
            has_psycopg3 = True
        except Exception:
            has_psycopg3 = False

        if not has_psycopg3:
            # Fall back to psycopg2 driver
            url = url.replace(prefix, "postgresql+psycopg2://", 1)
    return url

DATABASE_URL = _resolve_database_url(DATABASE_URL)
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

def get_db():
    """Get a database session. Use as a contex manager"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_connection():
    """Quick Connectivity Test"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version()"))
            print(f"Connected to: {result.fetchone()[0][:50]}...")
            return True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False
    
if __name__ == "__main__":
    test_connection()