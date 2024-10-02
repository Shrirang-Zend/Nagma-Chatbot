import pandas as pd
import re

def parse_release_date(date_str):
    """
    Parses a date string into a pandas Timestamp object.

    Parameters:
        date_str (str): The date string to parse.

    Returns:
        pd.Timestamp: Parsed date.
    """
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except ValueError:
        return pd.NaT

def format_release_date(date):
    """
    Formats a pandas Timestamp object into a human-readable date string.

    Parameters:
        date (pd.Timestamp): The date to format.

    Returns:
        str: Formatted date string.
    """
    if pd.notnull(date):
        if date.strftime('%m-%d') == '01-01':
            return date.strftime('%Y')
        else:
            return date.strftime('%Y-%m-%d')
    return 'Unknown'

def extract_numeric_range(text):
    """
    Extracts a numeric range from a text string.

    Parameters:
        text (str): The text to search.

    Returns:
        list: A list containing the lower and upper bounds of the range.
    """
    patterns = [
        r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)',
        r'between\s+(\d+\.?\d*)\s+and\s+(\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return sorted([float(match.group(1)), float(match.group(2))])
    return None