"""
Input/Output utility functions for the StockTrader application.

Contains helpers for file operations, compression, and data export.
"""
import io
import json
import hashlib
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Union, BinaryIO, Dict, Any, Tuple, Optional

import pandas as pd


def create_zip_archive(file_paths: List[Path]) -> bytes:
    """
    Creates a ZIP archive from a list of file paths.

    Args:
        file_paths: List of Path objects pointing to files to include in the archive

    Returns:
        Bytes object containing the ZIP archive data suitable for download
    """
    # Create an in-memory buffer
    zip_buffer = io.BytesIO()
    
    # Create a ZIP archive
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add each file to the archive
        for file_path in file_paths:
            if file_path.exists() and file_path.is_file():
                # Use the file's name as the archive name to avoid full paths
                zip_file.write(file_path, arcname=file_path.name)
    
    # Reset the buffer position to the start
    zip_buffer.seek(0)
    
    # Return the buffer's contents as bytes
    return zip_buffer.getvalue()


def save_dataframe(df: pd.DataFrame, path: Path, format: str = 'csv') -> Path:
    """
    Save a pandas DataFrame to a file with proper error handling.
    
    Args:
        df: Pandas DataFrame to save
        path: Path where the file should be saved
        format: File format (csv, pickle, etc.)
    
    Returns:
        Path to the saved file
    
    Raises:
        ValueError: If an unsupported format is specified
        OSError: If there are file permission issues
    """
    try:
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in the appropriate format
        if format.lower() == 'csv':
            df.to_csv(path, index=False)
        elif format.lower() == 'pickle' or format.lower() == 'pkl':
            df.to_pickle(path)
        elif format.lower() == 'excel' or format.lower() == 'xlsx':
            df.to_excel(path, index=False)
        elif format.lower() == 'json':
            df.to_json(path, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported file format: {format}")
            
        return path
    except Exception as e:
        # Re-raise with more context
        raise type(e)(f"Failed to save DataFrame to {path}: {str(e)}") from e


def safe_file_write(file_path: Path, content: str, create_backup: bool = True) -> Tuple[bool, str, Optional[Path]]:
    """
    Safely write file with backup and atomic operations.
    
    Args:
        file_path: Path where the file should be written
        content: String content to write
        create_backup: Whether to create a backup if file exists
    
    Returns:
        Tuple of (success, message, backup_path)
    """
    backup_path = None
    
    try:
        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists and backup requested
        if create_backup and file_path.exists():
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            shutil.copy2(file_path, backup_path)
        
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                       suffix=file_path.suffix,
                                       dir=file_path.parent) as tmp_file:
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        # Atomic move
        shutil.move(str(tmp_path), str(file_path))
        
        return True, "File written successfully", backup_path
        
    except Exception as e:
        return False, f"Failed to write file: {e}", backup_path


def save_dataframe_with_metadata(
    df: pd.DataFrame, 
    file_path: Path, 
    metadata: Dict[str, Any],
    create_backup: bool = True
) -> Tuple[bool, str, Optional[Path]]:
    """
    Save DataFrame with accompanying metadata file using atomic operations.
    
    Args:
        df: DataFrame to save
        file_path: Path for the CSV file
        metadata: Metadata dictionary to save as JSON
        create_backup: Whether to create backup of existing files
    
    Returns:
        Tuple of (success, message, backup_path)
    """
    backup_path = None
    
    try:
        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists and backup requested
        if create_backup and file_path.exists():
            backup_path = file_path.with_suffix('.csv.bak')
            shutil.copy2(file_path, backup_path)
        
        # Save DataFrame to temporary file first
        temp_csv_path = file_path.with_suffix('.tmp')
        df.to_csv(temp_csv_path, index=False)
        
        # Save metadata to temporary file
        metadata_path = file_path.with_suffix('.meta.json')
        temp_meta_path = metadata_path.with_suffix('.tmp')
        
        with open(temp_meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Atomic moves
        shutil.move(str(temp_csv_path), str(file_path))
        shutil.move(str(temp_meta_path), str(metadata_path))
        
        return True, f"Saved {len(df)} records with metadata", backup_path
        
    except Exception as e:
        # Clean up temporary files if they exist
        for temp_path in [file_path.with_suffix('.tmp'), 
                         file_path.with_suffix('.meta.json.tmp')]:
            if temp_path.exists():
                temp_path.unlink()
        
        return False, f"Failed to save data: {e}", backup_path


def calculate_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of a file for integrity checking.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (sha256, md5, etc.)
    
    Returns:
        Hex string of the file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def clean_directory(directory: Path, pattern: str = "*.csv", dry_run: bool = False) -> Tuple[int, List[str]]:
    """
    Clean files from a directory with optional dry run.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match (e.g., "*.csv", "*.tmp")
        dry_run: If True, only return what would be deleted
    
    Returns:
        Tuple of (count, list of file paths that were/would be removed)
    """
    if not directory.exists():
        return 0, []
    
    files_to_remove = list(directory.glob(pattern))
    removed_files = []
    
    for file_path in files_to_remove:
        try:
            if not dry_run:
                file_path.unlink()
            removed_files.append(str(file_path))
        except Exception:
            # Skip files that can't be removed
            continue
    
    return len(removed_files), removed_files


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Dictionary with file information
    """
    if not file_path.exists():
        return {"exists": False}
    
    stat = file_path.stat()
    
    return {
        "exists": True,
        "size": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "is_file": file_path.is_file(),
        "suffix": file_path.suffix,
        "name": file_path.name
    }


def load_dataframe_with_validation(file_path: Path, **pandas_kwargs) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load DataFrame with validation and error handling.
    
    Args:
        file_path: Path to the CSV file
        **pandas_kwargs: Additional arguments to pass to pd.read_csv
    
    Returns:
        Tuple of (DataFrame or None, status message)
    """
    try:
        if not file_path.exists():
            return None, f"File not found: {file_path}"
        
        # Check file size
        file_info = get_file_info(file_path)
        if file_info["size"] == 0:
            return None, "File is empty"
        
        # Load with default timestamp parsing
        default_kwargs = {
            'parse_dates': ['timestamp'] if 'parse_dates' not in pandas_kwargs else pandas_kwargs['parse_dates']
        }
        default_kwargs.update(pandas_kwargs)
        
        df = pd.read_csv(file_path, **default_kwargs)
        
        if df.empty:
            return None, "DataFrame is empty after loading"
        
        return df, f"Successfully loaded {len(df)} records"
        
    except Exception as e:
        return None, f"Failed to load file: {e}"


def load_metadata(file_path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Load metadata JSON file associated with a data file.
    
    Args:
        file_path: Path to the main data file (metadata will be .meta.json)
    
    Returns:
        Tuple of (metadata dict or None, status message)
    """
    metadata_path = file_path.with_suffix('.meta.json')
    
    try:
        if not metadata_path.exists():
            return None, "No metadata file found"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata, "Metadata loaded successfully"
        
    except Exception as e:
        return None, f"Failed to load metadata: {e}"


def export_session_data(
    symbols: List[str],
    date_range: Dict[str, str],
    interval: str,
    session_stats: Dict[str, Any]
) -> str:
    """
    Export session information as JSON string.
    
    Args:
        symbols: List of symbols processed
        date_range: Dictionary with start and end dates
        interval: Data interval used
        session_stats: Session statistics
    
    Returns:
        JSON string of session information
    """
    session_info = {
        'timestamp': datetime.now().isoformat(),
        'symbols': symbols,
        'date_range': date_range,
        'interval': interval,
        'session_stats': session_stats,
        'metadata': {
            'export_version': '1.0',
            'platform': 'StockTrader Dashboard'
        }
    }
    
    return json.dumps(session_info, indent=2, default=str)


def validate_file_path(file_path: Path, allowed_directory: Path) -> bool:
    """
    Validate that a file path is within an allowed directory (security check).
    
    Args:
        file_path: Path to validate
        allowed_directory: Directory that the path must be within
    
    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Resolve both paths to handle relative paths and symlinks
        resolved_file = file_path.resolve()
        resolved_dir = allowed_directory.resolve()
        
        # Check if the file path is within the allowed directory
        return resolved_file.is_relative_to(resolved_dir)
        
    except (OSError, ValueError):
        return False