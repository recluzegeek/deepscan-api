from sqlalchemy import create_engine
from databases import Database
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ModuleNotFoundError: No module named 'MySQLdb'
## `pip install mysqlclient`
SQLALCHEMY_DATABASE_URL = "mysql+mysqldb://admin:admin@localhost/deepscan"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
database = Database(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def execute_query(query, values=None):
    async with database:
        result = await database.fetch_all(query=query, values=values)
        return result
