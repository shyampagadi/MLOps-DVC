# src/data_preprocessing.py

import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from nltk.data import find

# ————————————————————————————————————————————————
# CONFIGURE NLTK DATA PATH
# ————————————————————————————————————————————————
nltk_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_dir)  # try custom first

# ————————————————————————————————————————————————
# SETUP LOGGER
# ————————————————————————————————————————————————
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
for handler in (logging.StreamHandler(), logging.FileHandler(os.path.join(log_dir, 'data_preprocessing.log'))):
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

# ————————————————————————————————————————————————
# DOWNLOAD & VERIFY NLTK PACKAGES
# ————————————————————————————————————————————————
def ensure_nltk(pkg, resource_path):
    try:
        find(resource_path)
        logger.info(f"NLTK resource '{pkg}' already present.")
    except LookupError:
        logger.info(f"Downloading NLTK resource '{pkg}'...")
        nltk.download(pkg, download_dir=nltk_dir, quiet=True)
        find(resource_path)  # will error if still missing
        logger.info(f"Downloaded and verified '{pkg}'.")

# Only download the core punkt (sentence + word tokenizer) and stopwords
ensure_nltk('punkt',     'tokenizers/punkt/english.pickle')    # core tokenizer :contentReference[oaicite:2]{index=2}
ensure_nltk('stopwords', 'corpora/stopwords')                  # stopwords list :contentReference[oaicite:3]{index=3}

# ————————————————————————————————————————————————
# TEXT TRANSFORM FUNCTIONS
# ————————————————————————————————————————————————
def transform_text(text: str) -> str:
    try:
        ps = PorterStemmer()
        tokens = nltk.word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalnum()]
        tokens = [t for t in tokens if t not in stopwords.words('english')]
        return " ".join(ps.stem(t) for t in tokens)
    except Exception as e:
        logger.error(f"transform_text error: {e}")
        raise

def preprocess_df(df: pd.DataFrame,
                  text_column: str = 'text',
                  target_column: str = 'target') -> pd.DataFrame:
    try:
        logger.debug("Starting preprocess_df()")
        df[target_column] = LabelEncoder().fit_transform(df[target_column])
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        logger.debug(f"Dropped {before - len(df)} duplicates")
        df[text_column] = df[text_column].apply(transform_text)
        logger.debug("Text transformation complete")
        return df
    except Exception as e:
        logger.error(f"preprocess_df error: {e}")
        raise

# ————————————————————————————————————————————————
# MAIN ENTRYPOINT
# ————————————————————————————————————————————————
def main():
    logger.info("Running main preprocessing pipeline")
    raw_dir    = os.path.join('data', 'raw')
    interim_dir = os.path.join('data', 'interim')
    os.makedirs(interim_dir, exist_ok=True)

    train_path = os.path.join(raw_dir, 'train.csv')
    test_path  = os.path.join(raw_dir, 'test.csv')
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Missing train.csv or test.csv in data/raw")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_out = preprocess_df(train_df)
    test_out  = preprocess_df(test_df)

    train_out.to_csv(os.path.join(interim_dir, 'train_processed.csv'), index=False)
    test_out.to_csv(os.path.join(interim_dir, 'test_processed.csv'),  index=False)
    logger.info(f"Preprocessed data saved to {interim_dir}")

if __name__ == "__main__":
    main()
