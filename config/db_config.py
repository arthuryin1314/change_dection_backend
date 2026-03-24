import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# database_url = "postgresql+asyncpg://postgres:123456@localhost/change_detection"

database_url=os.environ.get('DATABASE_URL')
if not database_url:
    raise RuntimeError("database_url environment variable is required")
async_engine = create_async_engine(
    database_url,
    # echo=True,
    echo=os.environ.get("SQL_ECHO", "false").lower() == "true",
    pool_size=10,  # Set the connection pool size
    max_overflow=20,  # Allow overflow connections beyond the pool size
    pool_pre_ping=True,  # Enable connection pre-ping to check if connections are alive
)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
    class_=AsyncSession
)
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise