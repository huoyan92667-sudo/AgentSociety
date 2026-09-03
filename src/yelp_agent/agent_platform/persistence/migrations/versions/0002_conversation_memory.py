"""add generic working memory and conversation episodes

Revision ID: 0002_conversation_memory
Revises: 0001_agent_persistence
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_conversation_memory"
down_revision: str | None = "0001_agent_persistence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = postgresql.JSONB(astext_type=sa.Text())
    op.create_table(
        "agent_working_memories",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("memory_json", json_type, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["agent_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_table(
        "agent_conversation_episodes",
        sa.Column("episode_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("topic", sa.String(length=300), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("details_json", json_type, nullable=False),
        sa.Column("source_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["agent_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("episode_id"),
    )
    op.create_index(
        "ix_agent_episodes_session_created",
        "agent_conversation_episodes",
        ["session_id", "created_at"],
    )
    op.create_index(
        "ix_agent_episodes_user_created",
        "agent_conversation_episodes",
        ["user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_episodes_user_created",
        table_name="agent_conversation_episodes",
    )
    op.drop_index(
        "ix_agent_episodes_session_created",
        table_name="agent_conversation_episodes",
    )
    op.drop_table("agent_conversation_episodes")
    op.drop_table("agent_working_memories")
