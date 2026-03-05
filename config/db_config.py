from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

database_url = "postgresql+asyncpg://postgres:123456@localhost/change_detection"

async_engine = create_async_engine(
    database_url,
    echo=True,  # Enable SQL query logging
    pool_size=10,  # Set the connection pool size
    max_overflow=20,  # Allow overflow connections beyond the pool size
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
        finally:
             await session.close()
