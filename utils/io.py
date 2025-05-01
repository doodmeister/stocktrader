"""
Input/Output utility functions for the StockTrader application.

Contains helpers for file operations, compression, and data export.
"""
import io
import zipfile
from pathlib import Path
from typing import List, Union, BinaryIO


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


def save_dataframe(df, path: Path, format: str = 'csv') -> Path:
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
            df.to_csv(path)
        elif format.lower() == 'pickle' or format.lower() == 'pkl':
            df.to_pickle(path)
        elif format.lower() == 'excel' or format.lower() == 'xlsx':
            df.to_excel(path)
        elif format.lower() == 'json':
            df.to_json(path)
        else:
            raise ValueError(f"Unsupported file format: {format}")
            
        return path
    except Exception as e:
        # Re-raise with more context
        raise type(e)(f"Failed to save DataFrame to {path}: {str(e)}") from e