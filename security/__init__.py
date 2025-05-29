"""
Security Package for StockTrader Platform

This package provides comprehensive security functionality including:
- Authentication and session management
- Authorization and access control
- Encryption and hashing utilities
- Input validation and sanitization

Usage:
    from security import authenticate, encrypt, authorize, utils
    
    # Or import specific functions
    from security.authentication import validate_session_security
    from security.encryption import create_secure_token
    from security.utils import sanitize_input
"""

# Import main security functions for convenient access
from .authentication import (
    validate_session_security,
    get_api_credentials,
    validate_credentials,
    get_sandbox_mode,
    get_openai_api_key,
    validate_api_key
)

from .authorization import (
    check_access_permission,
    validate_resource_access,
    get_user_permissions
)

from .encryption import (
    create_secure_token,
    generate_session_hash,
    validate_session_token,
    hash_password,
    verify_password,
    calculate_file_checksum,
    verify_file_checksum,
    generate_secure_token,
    create_data_signature,
    verify_data_signature
)

from .utils import (
    sanitize_input,
    validate_file_path,
    validate_input_length,
    prevent_path_traversal,
    sanitize_user_input,
    validate_file_size,
    validate_mime_type,
    generate_secure_filename,
    validate_json_structure
)

__version__ = "1.0.0"
__author__ = "StockTrader Security Team"

# Public API
__all__ = [
    # Authentication
    'validate_session_security',
    'get_api_credentials', 
    'validate_credentials',
    'get_sandbox_mode',
    'get_openai_api_key',
    'validate_api_key',
    
    # Authorization
    'check_access_permission',
    'validate_resource_access',
    'get_user_permissions',
      # Encryption
    'create_secure_token',
    'generate_session_hash',
    'validate_session_token',
    'hash_password',
    'verify_password',
    'calculate_file_checksum',
    'verify_file_checksum',
    'generate_secure_token',
    'create_data_signature',
    'verify_data_signature',
    
    # Utils
    'sanitize_input',
    'validate_file_path',
    'validate_input_length',
    'prevent_path_traversal',
    'sanitize_user_input',
    'validate_file_size',
    'validate_mime_type',
    'generate_secure_filename',
    'validate_json_structure'
]
