
from kedro.pipeline import node,Pipeline
from sentiment_analysis.pipelines.data_engineering.nodes import extract_small_big_corpus

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func = extract_small_big_corpus,
                inputs="raw_corpus",
                outputs=["small_corpus","big_corpus"],
                name="generate_corpuses",
            )
        ]
    )