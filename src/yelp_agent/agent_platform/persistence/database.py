"""异步数据库连接和会话工厂。"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import Field, SecretStr, field_validator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from yelp_agent.models import StrictModel

from .tables import Base


class DatabaseSettings(StrictModel):
    """数据库连接配置；连接地址不会出现在普通日志或模型上下文里。"""

    url: SecretStr
    echo_sql: bool = False
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)

    @field_validator("url")
    @classmethod
    def require_supported_async_driver(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        if not raw.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
            raise ValueError(
                "database URL must use postgresql+asyncpg or sqlite+aiosqlite"
            )
        return value

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> DatabaseSettings:
        source = os.environ if environment is None else environment
        raw = source.get("AGENT_DATABASE_URL")
        if not raw:
            raise ValueError("AGENT_DATABASE_URL is required")
        return cls(url=SecretStr(raw))


class AgentDatabase:
    """统一持有连接池，并向持久化实现提供短生命周期数据库会话。"""

    def __init__(self, settings: DatabaseSettings) -> None:
        url = settings.url.get_secret_value()
        engine_options: dict[str, object] = {
            "echo": settings.echo_sql,
            "pool_pre_ping": True,
        }
        if not url.startswith("sqlite+"):
            engine_options.update(
                pool_size=settings.pool_size,
                max_overflow=settings.max_overflow,
            )
        self.engine: AsyncEngine = create_async_engine(url, **engine_options)
        self.sessions: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

    async def create_schema_for_tests(self) -> None:
        """只供测试创建临时表；生产环境必须通过 Alembic 升级。"""

        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def drop_schema_for_tests(self) -> None:
        """只供测试清理临时表。"""

        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        await self.engine.dispose()
