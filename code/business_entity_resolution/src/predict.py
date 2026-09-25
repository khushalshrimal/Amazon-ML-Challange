"""
Inference module for generating candidate_pairs.tsv and matching_results.tsv for test data.
"""

from pathlib import Path
from . import config

def generate_test_outputs(output_dir: Path = config.OUTPUT_DIR):
    """
    Generate matching_results.tsv and candidate_pairs.tsv.
    
    TODO: Ensure every S1 test entity has exactly one row.
    TODO: Format matched_entity_ids as tab-separated TSV with comma-separated IDs.
    """
    pass
