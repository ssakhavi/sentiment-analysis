import pandas as pd
from imblearn.under_sampling import RandomUnderSampler



def extract_small_big_corpus(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Generate the small and big corpus from the raw corpus data.

        Args:
            corpus: Source data.
        Returns:
            small_corpus
            big_corpus.

    """

    
    X = corpus_df.index.to_numpy()
    y = corpus_df.overall.tolist()

    rus = RandomUnderSampler(random_state=42,sampling_strategy={1:1500,2:500,3:500,4:500,5:1500})
    X_undersample, _ = rus.fit_resample(X.reshape(-1, 1),y)
    small_corpus_df = corpus_df.iloc[X_undersample[:,0].tolist()]

    big_corpus_df = corpus_df.sample(100000, random_state = 42)

    return small_corpus_df, big_corpus_df
