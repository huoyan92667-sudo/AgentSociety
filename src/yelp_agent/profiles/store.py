"""Read-only exact-cutoff access to frozen user-profile artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Self

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.models import LocationCenter
from yelp_agent.profiles.schema import (
    PREFERENCE_SIGNAL_SCHEMA,
    PROFILE_SNAPSHOT_SCHEMA,
    TASK_PROFILE_LINK_SCHEMA,
    PreferenceSignal,
    ProfileEvidenceSummary,
    UserProfileV1,
)


class UserProfileStoreError(RuntimeError):
    """Raised when frozen profile artifacts are incomplete or invalid."""


class UserProfileNotFound(UserProfileStoreError):
    """Raised instead of selecting a profile from the wrong cutoff."""


class UserProfileStore:
    """Expose exact task-time reads and hide normalized Parquet assembly."""

    def __init__(self, artifact_root: str | Path) -> None:
        root = Path(artifact_root)
        paths_and_schemas = {
            "profile_snapshots": (
                root / "profile_snapshots.parquet",
                PROFILE_SNAPSHOT_SCHEMA,
            ),
            "preference_signals": (
                root / "preference_signals.parquet",
                PREFERENCE_SIGNAL_SCHEMA,
            ),
            "task_profile_map": (
                root / "task_profile_map.parquet",
                TASK_PROFILE_LINK_SCHEMA,
            ),
        }
        for path, schema in paths_and_schemas.values():
            if not path.is_file():
                raise FileNotFoundError(f"User-profile artifact does not exist: {path}")
            try:
                actual = pq.ParquetFile(path).schema_arrow
            except (OSError, pa.ArrowException) as exc:
                raise UserProfileStoreError(
                    f"Could not read user-profile artifact: {path}"
                ) from exc
            if not actual.equals(schema, check_metadata=False):
                raise UserProfileStoreError(
                    f"User-profile artifact has an unexpected schema: {path}"
                )
        self._connection = duckdb.connect()
        for view_name, (path, _) in paths_and_schemas.items():
            self._connection.from_parquet(str(path)).create_view(view_name)
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise UserProfileStoreError("User-profile store is closed")

    def _profile_id_for_exact_cutoff(
        self,
        user_id: str,
        cutoff_time: datetime,
    ) -> str:
        if not user_id or user_id != user_id.strip():
            raise ValueError("user_id must be nonempty without surrounding whitespace")
        if not isinstance(cutoff_time, datetime):
            raise TypeError("cutoff_time must be a datetime")
        rows = self._connection.execute(
            """
            SELECT profile_id
            FROM profile_snapshots
            WHERE user_id = ? AND cutoff_time = ?
            """,
            [user_id, cutoff_time],
        ).fetchall()
        if len(rows) != 1:
            raise UserProfileNotFound(
                f"No exact user profile exists for {user_id!r} at {cutoff_time!s}"
            )
        return str(rows[0][0])

    def _assemble(self, profile_id: str) -> UserProfileV1:
        snapshot_rows = self._connection.execute(
            "SELECT * FROM profile_snapshots WHERE profile_id = ?",
            [profile_id],
        ).fetchall()
        if len(snapshot_rows) != 1:
            raise UserProfileStoreError("Profile ID does not resolve to one snapshot")
        snapshot = dict(
            zip(PROFILE_SNAPSHOT_SCHEMA.names, snapshot_rows[0], strict=True)
        )
        preference_rows = self._connection.execute(
            """
            SELECT *
            FROM preference_signals
            WHERE profile_id = ?
            ORDER BY kind, value
            """,
            [profile_id],
        ).fetchall()
        signals: list[PreferenceSignal] = []
        for row in preference_rows:
            values = dict(zip(PREFERENCE_SIGNAL_SCHEMA.names, row, strict=True))
            for key in ("profile_id", "user_id", "cutoff_time"):
                values.pop(key)
            signals.append(PreferenceSignal.model_validate(values))

        categories = [signal for signal in signals if signal.kind == "category"]
        aspects = [signal for signal in signals if signal.kind == "aspect"]
        prices = [signal for signal in signals if signal.kind == "price"]
        areas = [signal for signal in signals if signal.kind == "area"]
        if len(prices) > 1:
            raise UserProfileStoreError("Profile contains multiple price preferences")
        latitude = snapshot["location_latitude"]
        longitude = snapshot["location_longitude"]
        if (latitude is None) != (longitude is None):
            raise UserProfileStoreError(
                "Profile contains an incomplete location center"
            )
        location = (
            None
            if latitude is None
            else LocationCenter(latitude=float(latitude), longitude=float(longitude))
        )
        return UserProfileV1(
            profile_id=str(snapshot["profile_id"]),
            user_id=str(snapshot["user_id"]),
            cutoff_time=snapshot["cutoff_time"],
            history_length=int(snapshot["history_length"]),
            average_rating=float(snapshot["average_rating"]),
            rating_distribution={
                str(stars): int(snapshot[f"rating_{stars}_count"])
                for stars in range(1, 6)
            },
            category_preferences=[signal for signal in categories if signal.score > 0],
            category_dislikes=[signal for signal in categories if signal.score < 0],
            aspect_preferences=[signal for signal in aspects if signal.score > 0],
            aspect_dislikes=[signal for signal in aspects if signal.score < 0],
            price_preference=prices[0] if prices else None,
            frequent_areas=sorted(
                areas, key=lambda signal: (-signal.score, signal.value)
            ),
            location_center=location,
            reliability=float(snapshot["reliability"]),
            evidence_summary=ProfileEvidenceSummary(
                category_evidence_count=int(snapshot["category_evidence_count"]),
                aspect_evidence_count=int(snapshot["aspect_evidence_count"]),
                price_evidence_count=int(snapshot["price_evidence_count"]),
                area_evidence_count=int(snapshot["area_evidence_count"]),
                first_interaction=snapshot["first_interaction"],
                last_interaction=snapshot["last_interaction"],
            ),
            profile_version=str(snapshot["profile_version"]),
        )

    def get(self, user_id: str, cutoff_time: datetime) -> UserProfileV1:
        """Return only a snapshot whose user and cutoff exactly match."""

        self._require_open()
        return self._assemble(self._profile_id_for_exact_cutoff(user_id, cutoff_time))

    def latest(
        self,
        user_id: str,
        *,
        at_or_before: datetime | None = None,
    ) -> UserProfileV1:
        """Return the newest known profile for a live request, optionally time-bounded.

        Benchmark code should keep using ``get`` with an exact cutoff.  This method is
        for the online recommendation entry, where the caller only knows the user.
        """

        self._require_open()
        if not user_id or user_id != user_id.strip():
            raise ValueError("user_id must be nonempty without surrounding whitespace")
        if at_or_before is not None and not isinstance(at_or_before, datetime):
            raise TypeError("at_or_before must be a datetime")
        if at_or_before is None:
            rows = self._connection.execute(
                """
                SELECT profile_id
                FROM profile_snapshots
                WHERE user_id = ?
                ORDER BY cutoff_time DESC, profile_id
                LIMIT 1
                """,
                [user_id],
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT profile_id
                FROM profile_snapshots
                WHERE user_id = ? AND cutoff_time <= ?
                ORDER BY cutoff_time DESC, profile_id
                LIMIT 1
                """,
                [user_id, at_or_before],
            ).fetchall()
        if len(rows) != 1:
            raise UserProfileNotFound(f"No user profile exists for {user_id!r}")
        return self._assemble(str(rows[0][0]))

    def for_task(self, task_id: str) -> UserProfileV1:
        """Resolve one frozen task mapping without accepting ground truth."""

        self._require_open()
        if not task_id or task_id != task_id.strip():
            raise ValueError("task_id must be nonempty without surrounding whitespace")
        rows = self._connection.execute(
            "SELECT profile_id FROM task_profile_map WHERE task_id = ?",
            [task_id],
        ).fetchall()
        if len(rows) != 1:
            raise UserProfileNotFound(f"No profile mapping exists for task {task_id!r}")
        return self._assemble(str(rows[0][0]))
