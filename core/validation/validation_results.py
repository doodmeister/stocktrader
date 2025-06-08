from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd

class ValidationResult(BaseModel):
    """Represents the result of a validation operation."""
    is_valid: bool = Field(..., description="Indicates if the validation was successful.")
    errors: Optional[List[str]] = Field(None, description="A list of error messages if validation failed.")
    validated_data: Optional[Any] = Field(None, description="The validated data, if successful.")

class DataFrameValidationResult(ValidationResult):
    """Represents the result of a DataFrame validation, including error details per row/column."""
    error_details: Optional[Dict[Union[int, str], List[str]]] = Field(
        None, 
        description="Detailed errors, mapping row index or column name to a list of error messages."
    )
    validated_data: Optional[pd.DataFrame] = Field(None, description="The validated DataFrame, if successful.")
