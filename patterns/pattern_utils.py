# stocktrader/pattern_utils.py

import ast
import inspect
from pathlib import Path
from typing import List, Optional, Tuple, Callable

from patterns.patterns import CandlestickPatterns

# Path to your patterns module
PATTERNS_PATH = Path(__file__).resolve().parent / "patterns.py"

def backup_file(path: Path) -> Path:
    """
    Make a .bak copy of the given file and return its path.
    """
    bak = path.with_suffix(".bak")
    bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return bak

def read_patterns_file() -> str:
    """
    Return the full source of patterns.py.
    """
    return PATTERNS_PATH.read_text(encoding="utf-8")

def write_patterns_file(content: str) -> Optional[Exception]:
    """
    Overwrite patterns.py (backing up first). Returns Exception on failure.
    """
    try:
        backup_file(PATTERNS_PATH)
        PATTERNS_PATH.write_text(content, encoding="utf-8")
        return None
    except Exception as e:
        return e

def get_pattern_names():
    """Get list of available pattern names."""
    patterns_instance = CandlestickPatterns()
    return patterns_instance.get_pattern_names()

def get_pattern_method(pattern_name: str) -> Optional[Callable]:
    """
    Given a pattern name, return the corresponding is_<pattern> method or None.
    """
    method_name = f"is_{pattern_name.lower().replace(' ', '_')}"
    return getattr(CandlestickPatterns, method_name, None)

def get_pattern_source_and_doc(
    pattern_method: Callable
) -> Tuple[str, Optional[str]]:
    """
    Return the source code and docstring for a given pattern-detection method.
    """
    source = inspect.getsource(pattern_method)
    doc = inspect.getdoc(pattern_method)
    return source, doc

def validate_python_code(code: str) -> bool:
    """
    Basic syntax check via AST parsing.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def add_candlestick_pattern_features(df, patterns_instance=None):
    """
    Add candlestick pattern features to a DataFrame.
    
    Args:
        df: DataFrame with OHLC data
        patterns_instance: Optional CandlestickPatterns instance
        
    Returns:
        DataFrame with added pattern features
    """
    if patterns_instance is None:
        patterns_instance = CandlestickPatterns()
    
    # Ensure we have a copy to avoid modifying the original
    result_df = df.copy()
    
    # Get available patterns
    pattern_names = patterns_instance.get_pattern_names()
    
    # Add pattern detection results as features
    for pattern_name in pattern_names:
        try:
            # Get the pattern detection method
            method_name = f"is_{pattern_name.lower().replace(' ', '_')}"
            pattern_method = getattr(patterns_instance, method_name, None)
            
            if pattern_method and callable(pattern_method):
                # Apply pattern detection to each row
                feature_name = f"pattern_{pattern_name.lower().replace(' ', '_')}"
                result_df[feature_name] = result_df.apply(
                    lambda row: pattern_method(row.to_dict()) if len(row) >= 4 else False,
                    axis=1
                ).astype(int)  # Convert boolean to int (0/1)
        except Exception as e:
            # Skip patterns that fail - they might need specific data formats
            print(f"Warning: Could not add pattern {pattern_name}: {e}")
            continue
    
    return result_df