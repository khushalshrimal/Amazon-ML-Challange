"""
Conservative, deterministic text normalization layer for Business Entity Resolution.
"""

import re
import unicodedata
import pandas as pd

# Legal suffix dictionary for standardization
LEGAL_SUFFIXES = {
    r"\bcorporation\b": "inc",
    r"\bcorp\b": "inc",
    r"\bincorporated\b": "inc",
    r"\bllc\b": "inc",
    r"\bllp\b": "inc",
    r"\blimited\b": "ltd",
    r"\bprivate\b": "pvt",
}

# Common address token dictionary
ADDRESS_TOKENS = {
    r"\broad\b": "rd",
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
}

def unicode_clean(text: str) -> str:
    """Apply NFKD normalization and convert to ASCII where possible."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("utf-8")

def normalize_name(name: str) -> str:
    """
    Conservative business name normalization:
    - Lowercase & ASCII unicode normalization
    - Strip punctuation and extra spaces
    - Standardize '&' to 'and'
    """
    if not isinstance(name, str) or pd.isna(name):
        return ""
    name = unicode_clean(name).lower()
    name = re.sub(r"&", " and ", name)
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def normalize_address(address: str) -> str:
    """
    Conservative address normalization:
    - Lowercase & ASCII unicode normalization
    - Strip punctuation and extra spaces
    """
    if not isinstance(address, str) or pd.isna(address):
        return ""
    addr = unicode_clean(address).lower()
    addr = re.sub(r"[^\w\s]", " ", addr)
    addr = re.sub(r"\s+", " ", addr).strip()
    return addr

def normalize_country(country: str) -> str:
    """
    Normalize country strings preserving open-set handling.
    """
    if not isinstance(country, str) or pd.isna(country):
        return "UNKNOWN"
    return country.strip().upper()
