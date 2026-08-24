"""一条命令完成旧词语、评论向量、意思相近候选和两路合并。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yelp_agent.recommendation_v2.review_index import build_review_vector_index
from yelp_agent.review_rag.config import load_review_rag_config
from yelp_agent.semantic_embedding import (
    LocalEmbeddingEnvironment,
    LocalQwenEmbeddingEncoder,
)

from .builder import (
    DEFAULT_BUSINESS_FACT_SOURCE,
    DEFAULT_CONFIG_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REVIEW_SOURCE,
    build_keyword_review_candidates,
)
from .merge import merge_review_candidates
from .semantic_recall import DEFAULT_INDEX_ROOT, build_semantic_review_candidates

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REVIEW_RAG_CONFIG = _PROJECT_ROOT / "configs" / "review_rag.yaml"

# 这里只找“是否谈到某种用餐体验”，不要求评论先属于某一家指定商户。
# 评论向量本身不带这段说明；它只帮助模型正确理解14种特征的查询示例。
_ASPECT_RECALL_INSTRUCTION = (
    "Represent a restaurant-review search request for retrieving passages that "
    "discuss the stated dining-experience aspect. Match both favorable and "
    "unfavorable descriptions; relevance does not imply positive sentiment."
)


def run_review_candidate_pipeline(
    *,
    model_path: Path,
    python_executable: Path,
    device: str = "cuda",
    embedding_batch_size: int = 16,
    top_reviews_per_business_aspect: int = 20,
    force_keyword: bool = False,
    force_index: bool = False,
    force_semantic: bool = False,
    force_merge: bool = False,
) -> dict[str, object]:
    """依次构建四层结果；已校验且来源未变化的层会直接复用。"""

    keyword = build_keyword_review_candidates(
        DEFAULT_REVIEW_SOURCE,
        DEFAULT_BUSINESS_FACT_SOURCE,
        DEFAULT_CONFIG_ROOT,
        DEFAULT_OUTPUT_ROOT,
        force=force_keyword,
    )
    rag_config = load_review_rag_config(DEFAULT_REVIEW_RAG_CONFIG)
    semantic_config = rag_config.semantic_config().model_copy(
        update={
            "batch_size": embedding_batch_size,
            "query_instruction": _ASPECT_RECALL_INSTRUCTION,
        }
    )
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        semantic_config,
        LocalEmbeddingEnvironment(
            model_path=model_path,
            python_executable=python_executable,
            device=device,
        ),
    )
    try:
        review_index = build_review_vector_index(
            encoder,
            business_fact_source=DEFAULT_BUSINESS_FACT_SOURCE,
            output_root=DEFAULT_INDEX_ROOT,
            force=force_index,
        )
        semantic = build_semantic_review_candidates(
            encoder,
            DEFAULT_INDEX_ROOT,
            DEFAULT_CONFIG_ROOT,
            DEFAULT_OUTPUT_ROOT,
            top_reviews_per_business_aspect=top_reviews_per_business_aspect,
            force=force_semantic,
        )
    finally:
        encoder.close()
    merged = merge_review_candidates(
        DEFAULT_REVIEW_SOURCE,
        DEFAULT_OUTPUT_ROOT,
        force=force_merge,
    )
    return {
        "keyword": keyword.model_dump(mode="json"),
        "review_index": review_index.model_dump(mode="json"),
        "semantic": semantic.model_dump(mode="json"),
        "merged": merged.model_dump(mode="json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成餐饮评论的两路粗筛候选")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        choices=range(1, 65),
        default=16,
        metavar="1..64",
        help="本地模型一次处理的评论片段数；显存足够时可适当调大",
    )
    parser.add_argument("--top-reviews-per-business-aspect", type=int, default=20)
    parser.add_argument("--force-keyword", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--force-semantic", action="store_true")
    parser.add_argument("--force-merge", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_review_candidate_pipeline(
        model_path=args.model_path,
        python_executable=args.python_executable,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        top_reviews_per_business_aspect=args.top_reviews_per_business_aspect,
        force_keyword=args.force_keyword,
        force_index=args.force_index,
        force_semantic=args.force_semantic,
        force_merge=args.force_merge,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
