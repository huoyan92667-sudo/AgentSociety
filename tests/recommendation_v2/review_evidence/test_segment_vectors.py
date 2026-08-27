import numpy as np

from yelp_agent.recommendation_v2.review_evidence.segment_vectors import (
    ReviewSegmentVectorStore,
)


def test_local_vector_store_reads_unique_point_ids_in_requested_order(tmp_path) -> None:
    path = tmp_path / "segment_embeddings.npy"
    np.save(
        path,
        np.asarray(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            dtype=np.float16,
        ),
    )
    store = ReviewSegmentVectorStore(path)

    loaded = store.get_many([2, 0, 2])

    assert list(loaded) == [2, 0]
    assert loaded[2].dtype == np.float32
    assert np.allclose(loaded[2], [0.0, 1.0])
    assert np.allclose(loaded[0], [1.0, 0.0])
    store.close()
