"""IO Utilities

Helper functions for input/output operations.
"""

import io
import zipfile
from pathlib import Path
from typing import List

def create_zip_archive(paths: List[Path]) -> bytes:
    """Create a ZIP archive in-memory from a list of file paths."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in paths:
            if path.exists():
                zip_file.write(path, arcname=path.name)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()