"""创建通用 Agent 的持久化表。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_agent_persistence"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = postgresql.JSONB(astext_type=sa.Text())

    op.create_table(
        "agent_sessions",
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("active_turn_id", sa.String(length=64), nullable=True),
        sa.Column("last_event_seq", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index("ix_agent_sessions_user_id", "agent_sessions", ["user_id"])

    op.create_table(
        "agent_turns",
        sa.Column("turn_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_message", sa.Text(), nullable=False),
        sa.Column("request_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("used_tools", json_type, nullable=False),
        sa.Column("usage_json", json_type, nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["agent_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("turn_id"),
    )
    op.create_index("ix_agent_turns_session_id", "agent_turns", ["session_id"])
    op.create_index("ix_agent_turns_status", "agent_turns", ["status"])

    op.create_table(
        "agent_events",
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("turn_id", sa.String(length=64), nullable=True),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", json_type, nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["agent_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["turn_id"],
            ["agent_turns.turn_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint(
            "session_id",
            "seq",
            name="uq_agent_event_session_seq",
        ),
    )
    op.create_index("ix_agent_events_session_id", "agent_events", ["session_id"])
    op.create_index(
        "ix_agent_events_turn_seq",
        "agent_events",
        ["turn_id", "seq"],
    )

    op.create_table(
        "agent_domain_state_versions",
        sa.Column("state_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("domain", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("state_json", json_type, nullable=False),
        sa.Column("source_event_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["agent_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("state_id"),
        sa.UniqueConstraint(
            "session_id",
            "domain",
            "version",
            name="uq_agent_domain_state_version",
        ),
    )
    op.create_index(
        "ix_agent_domain_state_latest",
        "agent_domain_state_versions",
        ["session_id", "domain", "version"],
    )

    op.create_table(
        "agent_result_artifacts",
        sa.Column("result_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("turn_id", sa.String(length=64), nullable=True),
        sa.Column("kind", sa.String(length=128), nullable=False),
        sa.Column("summary_json", json_type, nullable=False),
        sa.Column("content_json", json_type, nullable=True),
        sa.Column("storage_uri", sa.Text(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["agent_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["turn_id"],
            ["agent_turns.turn_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("result_id"),
    )
    op.create_index(
        "ix_agent_results_session_created",
        "agent_result_artifacts",
        ["session_id", "created_at"],
    )
    op.create_index(
        "ix_agent_result_artifacts_kind",
        "agent_result_artifacts",
        ["kind"],
    )

    op.create_table(
        "agent_llm_calls",
        sa.Column("llm_call_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("turn_id", sa.String(length=64), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("purpose", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("provider_request_id", sa.String(length=128), nullable=True),
        sa.Column("tool_call_id", sa.String(length=128), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["agent_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["turn_id"],
            ["agent_turns.turn_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("llm_call_id"),
    )
    op.create_index(
        "ix_agent_llm_calls_session_id",
        "agent_llm_calls",
        ["session_id"],
    )
    op.create_index(
        "ix_agent_llm_calls_turn_step",
        "agent_llm_calls",
        ["turn_id", "step_index"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_llm_calls_turn_step", table_name="agent_llm_calls")
    op.drop_index("ix_agent_llm_calls_session_id", table_name="agent_llm_calls")
    op.drop_table("agent_llm_calls")
    op.drop_index(
        "ix_agent_result_artifacts_kind",
        table_name="agent_result_artifacts",
    )
    op.drop_index(
        "ix_agent_results_session_created",
        table_name="agent_result_artifacts",
    )
    op.drop_table("agent_result_artifacts")
    op.drop_index(
        "ix_agent_domain_state_latest",
        table_name="agent_domain_state_versions",
    )
    op.drop_table("agent_domain_state_versions")
    op.drop_index("ix_agent_events_turn_seq", table_name="agent_events")
    op.drop_index("ix_agent_events_session_id", table_name="agent_events")
    op.drop_table("agent_events")
    op.drop_index("ix_agent_turns_status", table_name="agent_turns")
    op.drop_index("ix_agent_turns_session_id", table_name="agent_turns")
    op.drop_table("agent_turns")
    op.drop_index("ix_agent_sessions_user_id", table_name="agent_sessions")
    op.drop_table("agent_sessions")
