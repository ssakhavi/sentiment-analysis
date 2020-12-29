from kedro.pipeline import node, Pipeline
from sentiment_analysis.pipelines.data_engineering.nodes import (
    extract_small_big_corpus,
    extract_sentiment_score_by_counting,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=extract_small_big_corpus,
                inputs="raw_corpus",
                outputs=["small_corpus", "big_corpus"],
                name="generate_corpuses",
            ),
            node(
                func=extract_sentiment_score_by_counting,
                inputs="small_corpus",
                outputs="small_corpus_with_sentiment",
                name="sentiment_score",
            ),
        ]
    )
