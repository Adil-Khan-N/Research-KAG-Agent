"""
Database connection and session management.
Single place to get a Postgres connection — import this everywhere.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql://kag:kag@localhost:5432/kagdb")

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