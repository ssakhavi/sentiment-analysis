import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
from nltk.corpus import opinion_lexicon
from nltk import tokenize
import string
import math
from pandarallel import pandarallel

tqdm.pandas()
pandarallel.initialize()


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

    rus = RandomUnderSampler(
        random_state=42, sampling_strategy={1: 1500, 2: 500, 3: 500, 4: 500, 5: 1500}
    )
    X_undersample, _ = rus.fit_resample(X.reshape(-1, 1), y)
    small_corpus_df = corpus_df.iloc[X_undersample[:, 0].tolist()]

    big_corpus_df = corpus_df.sample(100000, random_state=42)

    # Only return Review Text(Reviews) and Review Value (Rating)

    small_corpus_df = select_and_rename_columns(small_corpus_df)
    big_corpus_df = select_and_rename_columns(big_corpus_df)

    return small_corpus_df, big_corpus_df


def select_and_rename_columns(df):
    df = df[["overall", "reviewText"]]
    df = df.rename(columns={"overall": "rating", "reviewText": "reviews"})
    return df


def extract_sentiment_score_by_counting(df: pd.DataFrame) -> pd.DataFrame:

    df["sentiment_score"] = df.progress_apply(calculate_sentiment_score, axis=1)

    return df


def calculate_sentiment_score(series: pd.Series):
    review = series["reviews"]
    if pd.notna(review):
        sentences = tokenize.sent_tokenize(review)
        word_list = {}

        for i, sent in enumerate(sentences):
            word_list[i] = tokenize.word_tokenize(sent)

        sentence_score = {}
        overall_score = 0
        for key, words in word_list.items():
            score = 0
            for word in words:
                if word.lower() in opinion_lexicon.positive():
                    score += 1
                if word.lower() in opinion_lexicon.negative():
                    score -= 1
            sentence_score[key] = score
            overall_score += math.copysign(1, score)

        return overall_score
    else:
        return 0
