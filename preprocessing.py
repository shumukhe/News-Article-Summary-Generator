import re
import pandas as pd
import numpy as np
import spacy
from time import time
from typing import Iterator, List


def text_strip(column: pd.Series) -> Iterator[str]:
    """Cleans a pandas Series of text data."""
    for row in column:
        row = str(row).lower()
        row = re.sub(r"(\\t|\\r|\\n)", " ", row)
        row = re.sub(r"(__+|--+|~~+|\+\++|\.\.+)", " ", row)
        row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", " ", row)
        row = re.sub(r"(mailto:)", " ", row)
        row = re.sub(r"(\\x9\d)", " ", row)
        row = re.sub(r"([iI][nN][cC]\d+)", "INC_NUM", row)
        row = re.sub(r"([cC][mM]\d+)|([cC][hH][gG]\d+)", "CM_NUM", row)
        row = re.sub(r"(\.\s+|\-\s+|\:\s+)", " ", row)
        row = re.sub(r"(\s+.\s+)", " ", row)
        
        try:
            url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', row)
            if url:
                repl_url = url.group(3)
                row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, row)
        except Exception:
            pass
        
        row = re.sub(r"\s+", " ", row)
        row = re.sub(r"(\s+.\s+)", " ", row)

        yield row


def spacy_cleaning(text_generator: Iterator[str], is_summary=False, spacy_model="en_core_web_sm") -> List[str]:
    """Applies spaCy processing using pipeline for performance."""
    nlp = spacy.load(spacy_model, disable=['ner', 'parser'])

    t = time()
    cleaned = [
        ('_START_ ' + str(doc) + ' _END_') if is_summary else str(doc)
        for doc in nlp.pipe(text_generator, batch_size=5000, n_threads=-1)
    ]
    print(f"Time to clean {'summary' if is_summary else 'text'}: {round((time() - t) / 60, 2)} mins")
    return cleaned


def filter_by_length(texts: List[str], summaries: List[str], max_text_len=100, max_summary_len=15) -> pd.DataFrame:
    """Filters data based on word count thresholds and adds special tokens."""
    short_text, short_summary = [], []

    for txt, summ in zip(texts, summaries):
        if len(txt.split()) <= max_text_len and len(summ.split()) <= max_summary_len:
            short_text.append(txt)
            short_summary.append("sostok " + summ + " eostok")

    return pd.DataFrame({'text': short_text, 'summary': short_summary})


def preprocess_data(df: pd.DataFrame, text_col='text', summary_col='summary',
                    spacy_model="en_core_web_sm",
                    max_text_len=100, max_summary_len=15) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    cleaned_text_gen = text_strip(df[text_col])
    cleaned_summary_gen = text_strip(df[summary_col])

    cleaned_text = spacy_cleaning(cleaned_text_gen, is_summary=False, spacy_model=spacy_model)
    cleaned_summary = spacy_cleaning(cleaned_summary_gen, is_summary=True, spacy_model=spacy_model)

    df['cleaned_text'] = pd.Series(cleaned_text)
    df['cleaned_summary'] = pd.Series(cleaned_summary)

    filtered_df = filter_by_length(cleaned_text, cleaned_summary, max_text_len, max_summary_len)
    return filtered_df

