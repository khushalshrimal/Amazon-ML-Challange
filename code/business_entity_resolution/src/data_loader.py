"""
Data loading utilities for large TSV files using memory-aware chunking and explicit delimiter settings.
"""

import pandas as pd
from typing import Iterator, List, Optional
from . import config

def load_tsv(filepath: str, usecols: Optional[List[str]] = None, chunksize: Optional[int] = None):
    """
    Load a tab-separated file with explicit sep='\\t'.
    
    TODO: Implement optional chunked iterator reading for memory safety on multi-million row datasets.
    """
    if chunksize:
        return pd.read_csv(filepath, sep="\t", usecols=usecols, chunksize=chunksize, dtype=str)
    return pd.read_csv(filepath, sep="\t", usecols=usecols, dtype=str)

def load_ground_truth(filepath: str = config.TRAIN_GT_PATH) -> pd.DataFrame:
    """
    Load train ground truth mappings.
    """
    return load_tsv(filepath, usecols=["source1_entity_id", "matched_entity_ids"])
