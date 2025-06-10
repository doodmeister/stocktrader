from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator
import pandas as pd

class ValidationResult(BaseModel):
    """Represents the result of a validation operation."""
    is_valid: bool = Field(..., description="Indicates if the validation was successful.")
    errors: Optional[List[str]] = Field(None, description="A list of error messages if validation failed.")
    warnings: List[str] = Field(default_factory=list, description="A list of warning messages.")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details about the validation result.")

    def add_error(self, error_message: str):
        """Adds an error message to the validation result."""
        self.is_valid = False
        if self.errors is None:
            self.errors = []
        self.errors.append(error_message)

    def add_warning(self, warning_message: str):
        """Adds a warning message to the validation result."""
        if warning_message not in self.warnings:
            self.warnings.append(warning_message)

class DataFrameValidationResult(ValidationResult):
    """Represents the result of a DataFrame validation, including error details per row/column."""
    error_details: Optional[Dict[Union[int, str], List[str]]] = Field(
        None, 
        description="Detailed errors, mapping row index or column name to a list of error messages."
    )
    validated_data: Optional[pd.DataFrame] = Field(None, description="The validated DataFrame, if successful.")
    dataframe_shape: Optional[Tuple[int, int]] = None
    missing_columns: List[str] = Field(default_factory=list)
    duplicate_columns: List[str] = Field(default_factory=list)
    nan_counts: Optional[Dict[str, int]] = None
    data_type_issues: Optional[Dict[str, str]] = None # e.g. {"column_name": "Expected int, got float"}
    failed_rows_summary: Optional[pd.DataFrame] = None # A sample of rows that failed validation

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("failed_rows_summary")
    def validate_failed_rows_summary(cls, value):
        """Validates the failed rows summary field."""
        if value is not None and not isinstance(value, pd.DataFrame):
            raise ValueError("failed_rows_summary must be a pandas DataFrame")
        return value
