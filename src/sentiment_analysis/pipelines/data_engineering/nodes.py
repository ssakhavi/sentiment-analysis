from typing import List
from nltk.corpus.reader import wordlist
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
from nltk.corpus import opinion_lexicon
from nltk import tokenize
from nltk.sentiment.util import mark_negation
import string
import math
from pandarallel import pandarallel
from nltk.corpus import stopwords
import regex as re


tqdm.pandas()
pandarallel.initialize()

stopword_list = stopwords.words("english")
stopword_list.remove("not")


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


def remove_extra_whitespace_tabs(text):
    # pattern = r'^\s+$|\s+$'
    pattern = r"^\s*|\s\s*"
    return re.sub(pattern, " ", text).strip()


def preprocess_sentences(sents):
    word_list = {}

    if not isinstance(sents, list):
        sents = [sents]

    for i, sent in enumerate(sents):
        # Pre-Tokenization

        ## Remove Punctuations
        sent = sent.translate(str.maketrans("", "", string.punctuation))

        ## lower-case
        sent = sent.lower()

        ## Remove Extra White Spaces and Tabs
        sent = remove_extra_whitespace_tabs(sent)

        # Tokenization
        word_tokens = tokenize.word_tokenize(sent)

        # Post-Tokenization

        ## Remove stopwords
        word_tokens = [w for w in word_tokens if not w in stopword_list]

        ## Mark Negation
        word_tokens = mark_negation(word_tokens)

        word_list[i] = word_tokens
        ## Mark Negation

    return word_list


def calculate_sentiment_score(series: pd.Series):
    review = series["reviews"]
    if pd.notna(review):
        sentences = tokenize.sent_tokenize(review)
        word_list = preprocess_sentences(sentences)

        # print("\nPrinting Positive/Negative Words(if any)")
        sentence_score = {}
        overall_score = 0
        for key, words in word_list.items():
            score = 0
            for word in words:
                score_sign = 1
                if len(word.split("_")) > 1:
                    score_sign = -1 if word.split("_")[1] == "NEG" else 1
                if word.split("_")[0] in opinion_lexicon.positive():
                    # print(f"{word}: Pos")
                    score += 1 * score_sign
                if word.split("_")[0] in opinion_lexicon.negative():
                    # print(f"{word}: Neg")
                    score -= 1 * score_sign
            sentence_score[key] = score
            overall_score += math.copysign(1, score) if abs(score) > 0 else 0

        return overall_score
    else:
        return 0
