from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

# Pass credentials separately to avoid URL parsing issues with special characters.
# Allow overriding connection string via DATABASE_URL (useful for SQLite testing).
database_url = os.environ.get("DATABASE_URL")
if database_url:
    engine = create_engine(database_url)
else:
    engine = create_engine(
        "postgresql+psycopg2://",
        creator=lambda: __import__('psycopg2').connect(
            host="localhost",
            port=5432,
            dbname="Verity",
            user="postgres",
            password=os.environ["DB_PASSWORD"],
        )
    )


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()