"""把统一结构中的硬条件翻译成安全查询，并过滤真实餐厅事实。"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pyarrow as pa
from pydantic import Field, field_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.business_facts import (
    BusinessFact,
    BusinessFactCatalog,
    is_open_at,
    parse_visit_time,
)
from yelp_agent.recommendation_v2.category_catalog import FixedCategoryCatalog
from yelp_agent.recommendation_v2.schema import (
    DefaultConstraint,
    HardConstraint,
    HardConstraintField,
    HardOperator,
    RequirementValue,
    UnifiedRecommendationState,
)

from .geography import GeographicDistanceResult

type StructuredConstraint = HardConstraint | DefaultConstraint

_NUMERIC_COLUMNS: dict[HardConstraintField, str] = {
    "price_level": "f.price_level",
    "rating": "f.rating",
    "review_count": "f.review_count",
}
_BOOLEAN_COLUMNS: dict[HardConstraintField, str] = {
    "accepts_reservations": "f.accepts_reservations",
    "delivery": "f.delivery",
    "takeout": "f.takeout",
    "outdoor_seating": "f.outdoor_seating",
    "good_for_kids": "f.good_for_kids",
    "good_for_groups": "f.good_for_groups",
    "wheelchair_accessible": "f.wheelchair_accessible",
    "dogs_allowed": "f.dogs_allowed",
    "parking_available": "f.parking_available",
}
_SQL_OPERATORS: dict[HardOperator, str] = {
    "equals": "=",
    "less_than": "<",
    "less_than_or_equal": "<=",
    "greater_than": ">",
    "greater_than_or_equal": ">=",
}


class FilteredBusiness(StrictModel):
    """满足全部硬条件的一家餐厅及其本轮距离。"""

    business: BusinessFact
    distance_km: float | None = Field(default=None, ge=0)


class HardFilterStep(StrictModel):
    """一条硬条件执行前后的数量变化，便于核对到底过滤了什么。"""

    key: str = Field(min_length=1)
    field: HardConstraintField
    operator: HardOperator
    value: RequirementValue
    source: str = Field(min_length=1)
    before_count: int = Field(ge=0)
    after_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    unknown_excluded_count: int = Field(ge=0)
    sql_condition: str = Field(min_length=1)


class StructuredHardFilterResult(StrictModel):
    """结构化硬过滤的完整结果，既供后续排序使用，也保留查询审计信息。"""

    source_business_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    candidate_business_ids: list[str]
    candidates: list[FilteredBusiness]
    steps: list[HardFilterStep]
    generated_sql: str = Field(min_length=1)
    sql_parameters: list[RequirementValue]

    @field_validator("candidate_business_ids")
    @classmethod
    def validate_unique_candidates(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("filtered candidate business IDs must be unique")
        return values


class StructuredHardFilterTool:
    """只接受已经通过统一结构校验的硬条件，不解析用户自然语言。"""

    name = "apply_structured_hard_filter"

    def __init__(
        self,
        business_catalog: BusinessFactCatalog,
        category_catalog: FixedCategoryCatalog,
    ) -> None:
        self._businesses = business_catalog
        self._categories = category_catalog

    def execute(
        self,
        state: UnifiedRecommendationState,
        *,
        geography: GeographicDistanceResult | None = None,
        candidate_business_ids: list[str] | None = None,
    ) -> StructuredHardFilterResult:
        """按硬条件和仍然生效的场景默认条件过滤真实餐厅。"""

        constraints: list[StructuredConstraint] = [
            *state.hard_constraints,
            *state.default_constraints,
        ]
        needs_distance = any(item.field == "distance_km" for item in constraints)
        open_at_values = {
            str(item.value) for item in constraints if item.field == "open_at"
        }
        if len(open_at_values) > 1:
            raise ValueError("hard filter accepts one effective visit time")
        visit_time = (
            parse_visit_time(next(iter(open_at_values)))
            if open_at_values
            else None
        )
        if needs_distance and geography is None:
            raise ValueError("distance filtering requires geographic distances")
        if geography is not None and state.search_center != geography.search_center:
            raise ValueError("geographic distances use a different search center")

        scope = candidate_business_ids
        if scope is not None:
            if len(scope) != len(set(scope)):
                raise ValueError("candidate business IDs must be unique")
            for business_id in scope:
                self._businesses.get(business_id)

        with duckdb.connect(database=":memory:") as connection:
            self._register_sources(
                connection,
                geography=geography,
                scope=scope,
                visit_time=visit_time,
            )
            from_sql = self._from_sql(
                geography is not None,
                scope is not None,
                visit_time is not None,
            )
            clauses: list[str] = []
            parameters: list[RequirementValue] = []
            source_count = self._count(connection, from_sql, clauses, parameters)
            steps: list[HardFilterStep] = []

            for constraint in constraints:
                before_count = self._count(
                    connection,
                    from_sql,
                    clauses,
                    parameters,
                )
                sql_condition, condition_parameters, unknown_condition = (
                    self._constraint_sql(constraint)
                )
                unknown_count = (
                    self._count(
                        connection,
                        from_sql,
                        [*clauses, unknown_condition],
                        parameters,
                    )
                    if unknown_condition is not None
                    else 0
                )
                clauses.append(sql_condition)
                parameters.extend(condition_parameters)
                after_count = self._count(
                    connection,
                    from_sql,
                    clauses,
                    parameters,
                )
                steps.append(
                    HardFilterStep(
                        key=constraint.key,
                        field=constraint.field,
                        operator=constraint.operator,
                        value=constraint.value,
                        source=constraint.controlling_source,
                        before_count=before_count,
                        after_count=after_count,
                        excluded_count=before_count - after_count,
                        unknown_excluded_count=unknown_count,
                        sql_condition=sql_condition,
                    )
                )

            select_distance = (
                "d.distance_km" if geography is not None else "NULL::DOUBLE AS distance_km"
            )
            generated_sql = (
                f"SELECT f.*, {select_distance} FROM {from_sql}"
                + self._where_sql(clauses)
                + " ORDER BY f.business_id"
            )
            rows = (
                connection.execute(generated_sql, parameters)
                .to_arrow_table()
                .to_pylist()
            )

        candidates: list[FilteredBusiness] = []
        for row in rows:
            distance = row.pop("distance_km")
            candidates.append(
                FilteredBusiness(
                    business=BusinessFact.model_validate(row),
                    distance_km=distance,
                )
            )
        candidate_ids = [item.business.business_id for item in candidates]
        return StructuredHardFilterResult(
            source_business_count=source_count,
            candidate_count=len(candidates),
            candidate_business_ids=candidate_ids,
            candidates=candidates,
            steps=steps,
            generated_sql=generated_sql,
            sql_parameters=parameters,
        )

    def _register_sources(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        geography: GeographicDistanceResult | None,
        scope: list[str] | None,
        visit_time: datetime | None,
    ) -> None:
        """把已校验的事实文件和本轮工具结果注册成只读查询表。"""

        escaped_path = str(self._businesses.fact_path).replace("'", "''")
        connection.execute(
            "CREATE TEMP VIEW business_facts AS "
            f"SELECT * FROM read_parquet('{escaped_path}')"
        )
        if geography is not None:
            connection.register(
                "business_distances",
                pa.Table.from_pylist(
                    [item.model_dump() for item in geography.distances],
                    schema=pa.schema(
                        [
                            pa.field("business_id", pa.string(), nullable=False),
                            pa.field("distance_km", pa.float64(), nullable=False),
                        ]
                    ),
                ),
            )
        if scope is not None:
            connection.register(
                "candidate_scope",
                pa.Table.from_pylist(
                    [{"business_id": item} for item in scope],
                    schema=pa.schema(
                        [pa.field("business_id", pa.string(), nullable=False)]
                    ),
                ),
            )
        if visit_time is not None:
            connection.register(
                "business_open_status",
                pa.Table.from_pylist(
                    [
                        {
                            "business_id": business.business_id,
                            "is_open_at": is_open_at(business, visit_time),
                        }
                        for business in self._businesses.all()
                    ],
                    schema=pa.schema(
                        [
                            pa.field("business_id", pa.string(), nullable=False),
                            pa.field("is_open_at", pa.bool_(), nullable=True),
                        ]
                    ),
                ),
            )

    @staticmethod
    def _from_sql(
        has_geography: bool,
        has_scope: bool,
        has_open_status: bool,
    ) -> str:
        joins = ["business_facts f"]
        if has_geography:
            joins.append("JOIN business_distances d USING (business_id)")
        if has_scope:
            joins.append("JOIN candidate_scope s USING (business_id)")
        if has_open_status:
            joins.append("JOIN business_open_status o USING (business_id)")
        return " ".join(joins)

    @staticmethod
    def _where_sql(clauses: list[str]) -> str:
        return (
            ""
            if not clauses
            else " WHERE " + " AND ".join(f"({item})" for item in clauses)
        )

    @classmethod
    def _count(
        cls,
        connection: duckdb.DuckDBPyConnection,
        from_sql: str,
        clauses: list[str],
        parameters: list[RequirementValue],
    ) -> int:
        query = f"SELECT COUNT(*) FROM {from_sql}" + cls._where_sql(clauses)
        row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise RuntimeError("hard-filter count query returned no row")
        return int(row[0])

    def _constraint_sql(
        self,
        constraint: StructuredConstraint,
    ) -> tuple[str, list[RequirementValue], str | None]:
        """从封闭字段表生成参数化条件；用户内容永远不会拼进查询语句。"""

        if constraint.field == "category":
            return self._category_sql(constraint)
        if constraint.field == "business_id":
            values = list(constraint.value)  # type: ignore[arg-type]
            present = "list_contains(?::VARCHAR[], f.business_id)"
            if constraint.operator == "any_of":
                return present, [values], None
            if constraint.operator == "none_of":
                return f"NOT {present}", [values], None
            # 商家编号是单值事实，因此 all_of 只有一个目标时才可能命中。
            if len(values) == 1:
                return present, [values], None
            return "FALSE", [], None
        if constraint.field == "distance_km":
            operator = _SQL_OPERATORS[constraint.operator]
            return (
                f"d.distance_km IS NOT NULL AND d.distance_km {operator} ?",
                [constraint.value],
                "d.distance_km IS NULL",
            )
        if constraint.field == "open_at":
            return (
                "o.is_open_at IS TRUE",
                [],
                "o.is_open_at IS NULL",
            )
        if constraint.field in _BOOLEAN_COLUMNS:
            column = _BOOLEAN_COLUMNS[constraint.field]
            return (
                f"{column} IS NOT NULL AND {column} = ?",
                [constraint.value],
                f"{column} IS NULL",
            )
        column = _NUMERIC_COLUMNS[constraint.field]
        operator = _SQL_OPERATORS[constraint.operator]
        return (
            f"{column} IS NOT NULL AND {column} {operator} ?",
            [constraint.value],
            f"{column} IS NULL",
        )

    def _category_sql(
        self,
        constraint: StructuredConstraint,
    ) -> tuple[str, list[RequirementValue], None]:
        requested = list(constraint.value)  # type: ignore[arg-type]
        expanded = [
            list(self._categories.expand_for_filter(item)) for item in requested
        ]
        conditions = ["list_has_any(f.categories, ?::VARCHAR[])" for _ in expanded]
        if constraint.operator == "any_of":
            clause = " OR ".join(conditions)
        elif constraint.operator == "all_of":
            clause = " AND ".join(conditions)
        else:
            clause = "NOT (" + " OR ".join(conditions) + ")"
        return clause, expanded, None
