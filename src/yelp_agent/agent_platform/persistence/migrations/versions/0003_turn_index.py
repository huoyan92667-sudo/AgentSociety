"""add a stable per-session turn index

Revision ID: 0003_turn_index
Revises: 0002_conversation_memory
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_turn_index"
down_revision: str | None = "0002_conversation_memory"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """给已有轮次补出稳定顺序，再开启非空和唯一约束。"""

    op.add_column(
        "agent_sessions",
        sa.Column(
            "last_turn_index",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "agent_turns",
        sa.Column("turn_index", sa.Integer(), nullable=True),
    )
    op.execute(
        sa.text(
            """
            WITH numbered AS (
                SELECT
                    turn_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY session_id
                        ORDER BY started_at, request_time, turn_id
                    ) AS value
                FROM agent_turns
            )
            UPDATE agent_turns AS target
            SET turn_index = numbered.value
            FROM numbered
            WHERE target.turn_id = numbered.turn_id
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE agent_sessions AS target
            SET last_turn_index = source.value
            FROM (
                SELECT session_id, MAX(turn_index) AS value
                FROM agent_turns
                GROUP BY session_id
            ) AS source
            WHERE target.session_id = source.session_id
            """
        )
    )
    op.alter_column("agent_turns", "turn_index", nullable=False)
    op.create_unique_constraint(
        "uq_agent_turn_session_index",
        "agent_turns",
        ["session_id", "turn_index"],
    )
    # 0002 已经写入过的摘要水位只有时间。这里把它一次性换算为轮次编号，
    # 避免升级后重复把旧原文放回模型上下文。
    op.execute(
        sa.text(
            """
            UPDATE agent_working_memories AS memory
            SET memory_json = jsonb_set(
                memory.memory_json,
                '{summarized_through_turn_index}',
                to_jsonb(boundary.turn_index),
                true
            )
            FROM (
                SELECT
                    stored.session_id,
                    MAX(turns.turn_index) AS turn_index
                FROM agent_working_memories AS stored
                JOIN agent_turns AS turns
                  ON turns.session_id = stored.session_id
                WHERE stored.memory_json ? 'summarized_through'
                  AND stored.memory_json ->> 'summarized_through' IS NOT NULL
                  AND turns.ended_at <= (
                      stored.memory_json ->> 'summarized_through'
                  )::timestamptz
                GROUP BY stored.session_id
            ) AS boundary
            WHERE memory.session_id = boundary.session_id
            """
        )
    )
    op.alter_column("agent_sessions", "last_turn_index", server_default=None)


def downgrade() -> None:
    op.drop_constraint(
        "uq_agent_turn_session_index",
        "agent_turns",
        type_="unique",
    )
    op.drop_column("agent_turns", "turn_index")
    op.drop_column("agent_sessions", "last_turn_index")
