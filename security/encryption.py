"""
Encryption Module for StockTrader Security Package

Handles cryptographic operations including hashing, token generation,
password management, and data encryption/decryption.
"""

import hashlib
import secrets
import base64
import logging
import time
import hmac
from pathlib import Path
from typing import Union
from typing import Optional, Tuple
import streamlit as st

logger = logging.getLogger(__name__)


def create_secure_token(length: int = 32) -> str:
    """
    Create a cryptographically secure random token.
    
    Args:
        length: Length of the token in bytes (default 32)
        
    Returns:
        str: Secure hexadecimal token
    """
    return secrets.token_hex(length)


def create_url_safe_token(length: int = 32) -> str:
    """
    Create a URL-safe random token.
    
    Args:
        length: Length of the token in bytes (default 32)
        
    Returns:
        str: URL-safe token
    """
    return secrets.token_urlsafe(length)


def generate_session_hash(data: str, salt: Optional[str] = None) -> str:
    """
    Generate a secure hash for session data.
    
    Args:
        data: Data to hash
        salt: Optional salt for the hash
        
    Returns:
        str: Hexadecimal hash string
    """
    if salt is None:
        salt = ""
    
    # Combine data with salt
    salted_data = f"{data}{salt}"
    
    # Generate SHA-256 hash
    return hashlib.sha256(salted_data.encode('utf-8')).hexdigest()


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password with a salt.
    
    Args:
        password: Password to hash
        salt: Optional salt (will generate if not provided)
        
    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = create_secure_token(16)  # 16 bytes = 32 hex chars
    
    # Combine password with salt
    salted_password = f"{password}{salt}"
    
    # Generate hash using PBKDF2 (more secure than simple SHA-256)
    try:
        import hashlib
        hashed = hashlib.pbkdf2_hmac('sha256', 
                                   salted_password.encode('utf-8'), 
                                   salt.encode('utf-8'), 
                                   100000)  # 100,000 iterations
        return hashed.hex(), salt
    except Exception:
        # Fallback to SHA-256 if PBKDF2 is not available
        hashed = hashlib.sha256(salted_password.encode('utf-8')).hexdigest()
        return hashed, salt


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        password: Plain text password to verify
        hashed_password: Stored hash to compare against
        salt: Salt used for the original hash
        
    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        # Hash the provided password with the same salt
        computed_hash, _ = hash_password(password, salt)
        
        # Compare hashes using constant-time comparison
        return secrets.compare_digest(computed_hash, hashed_password)
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False


def validate_session_token(token: str) -> bool:
    """
    Validate a session token format and existence.
    
    Args:
        token: Token to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not token:
        return False
    
    # Check token format (should be hex string)
    try:
        int(token, 16)
    except ValueError:
        return False
    
    # Check token length (64 chars = 32 bytes hex)
    if len(token) != 64:
        return False
    
    # Check if token exists in session state
    if hasattr(st, 'session_state'):
        session_token = getattr(st.session_state, 'session_token', None)
        return session_token == token
    
    return False


def encrypt_data(data: str, key: Optional[str] = None) -> Tuple[str, str]:
    """
    Encrypt data using a simple XOR cipher (for demonstration purposes).
    
    Note: This is a basic implementation. For production use, consider
    using a more robust encryption library like cryptography.
    
    Args:
        data: Data to encrypt
        key: Encryption key (will generate if not provided)
        
    Returns:
        Tuple of (encrypted_data_base64, key)
    """
    if key is None:
        key = create_secure_token(16)
    
    # Simple XOR encryption (for demo purposes)
    key_bytes = key.encode('utf-8')
    data_bytes = data.encode('utf-8')
    
    encrypted_bytes = bytearray()
    for i, byte in enumerate(data_bytes):
        encrypted_bytes.append(byte ^ key_bytes[i % len(key_bytes)])
    
    # Encode as base64 for safe storage/transmission
    encrypted_base64 = base64.b64encode(encrypted_bytes).decode('utf-8')
    
    return encrypted_base64, key


def decrypt_data(encrypted_data_base64: str, key: str) -> str:
    """
    Decrypt data encrypted with encrypt_data().
    
    Args:
        encrypted_data_base64: Base64 encoded encrypted data
        key: Decryption key
        
    Returns:
        str: Decrypted data
    """
    try:
        # Decode from base64
        encrypted_bytes = base64.b64decode(encrypted_data_base64.encode('utf-8'))
        
        # XOR decryption (same as encryption for XOR)
        key_bytes = key.encode('utf-8')
        decrypted_bytes = bytearray()
        
        for i, byte in enumerate(encrypted_bytes):
            decrypted_bytes.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return decrypted_bytes.decode('utf-8')
    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        return ""


def generate_api_key() -> str:
    """
    Generate a new API key.
    
    Returns:
        str: Generated API key
    """
    # Generate a 32-byte key and encode as base64
    key_bytes = secrets.token_bytes(32)
    api_key = base64.b64encode(key_bytes).decode('utf-8')
    
    # Remove padding for cleaner appearance
    return api_key.rstrip('=')


def hash_file_content(file_path: str) -> str:
    """
    Generate a hash of file content for integrity checking.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: SHA-256 hash of the file content
    """
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256()
            
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
            
            return file_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        return ""


def constant_time_compare(a: str, b: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        bool: True if strings are equal, False otherwise
    """
    return secrets.compare_digest(a, b)


def generate_checksum(data: Union[str, bytes]) -> str:
    """
    Generate a checksum for data integrity verification.
    
    Args:
        data: Data to generate checksum for
        
    Returns:
        str: MD5 checksum (for speed, not security)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.md5(data).hexdigest()


def verify_checksum(data: Union[str, bytes], expected_checksum: str) -> bool:
    """
    Verify data integrity using checksum.
    
    Args:
        data: Data to verify
        expected_checksum: Expected checksum value
        
    Returns:
        bool: True if checksum matches, False otherwise
    """
    actual_checksum = generate_checksum(data)
    return constant_time_compare(actual_checksum, expected_checksum)


def secure_random_string(length: int = 16, 
                        include_uppercase: bool = True,
                        include_lowercase: bool = True,
                        include_digits: bool = True,
                        include_symbols: bool = False) -> str:
    """
    Generate a secure random string with specified character set.
    
    Args:
        length: Length of the string
        include_uppercase: Include uppercase letters
        include_lowercase: Include lowercase letters
        include_digits: Include digits
        include_symbols: Include symbols
        
    Returns:
        str: Secure random string
    """
    import string
    
    characters = ""
    if include_uppercase:
        characters += string.ascii_uppercase
    if include_lowercase:
        characters += string.ascii_lowercase
    if include_digits:
        characters += string.digits
    if include_symbols:
        characters += "!@#$%^&*"
    
    if not characters:
        raise ValueError("At least one character type must be included")
    
    return ''.join(secrets.choice(characters) for _ in range(length))


def calculate_file_checksum(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Calculate checksum for file integrity verification.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use ('sha256', 'md5', 'sha1')
    
    Returns:
        Hexadecimal digest of the file checksum
    """
    import hashlib
    
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hash_obj = hashlib.new(algorithm.lower())
        
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate checksum for {file_path}: {e}")
        return ""


def verify_file_checksum(file_path: Union[str, Path], expected_checksum: str, 
                        algorithm: str = 'sha256') -> bool:
    """
    Verify file integrity using checksum comparison.
    
    Args:
        file_path: Path to the file to verify
        expected_checksum: Expected checksum value
        algorithm: Hash algorithm used for verification
    
    Returns:
        True if checksums match, False otherwise
    """
    if not expected_checksum:
        logger.warning("No expected checksum provided for verification")
        return True  # No checksum to verify
    
    try:
        actual_checksum = calculate_file_checksum(file_path, algorithm)
        match = actual_checksum.lower() == expected_checksum.lower()
        
        if not match:
            logger.warning(f"Checksum mismatch for {file_path}")
            logger.debug(f"Expected: {expected_checksum}, Actual: {actual_checksum}")
        
        return match
    except Exception as e:
        logger.error(f"Checksum verification failed for {file_path}: {e}")
        return False


def generate_secure_token(length: int = 32, url_safe: bool = True) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Length of the token in bytes
        url_safe: Whether to generate URL-safe token
    
    Returns:
        Secure random token string
    """
    try:
        if url_safe:
            return secrets.token_urlsafe(length)
        else:
            return secrets.token_hex(length)
    except Exception as e:
        logger.error(f"Token generation failed: {e}")
        # Fallback to basic token generation
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:length*2]


def create_data_signature(data: str, secret_key: str) -> str:
    """
    Create HMAC signature for data integrity verification.
    
    Args:
        data: Data to sign
        secret_key: Secret key for signing
    
    Returns:
        HMAC signature in hexadecimal format
    """
    import hmac
    
    try:
        signature = hmac.new(
            secret_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        )
        return signature.hexdigest()
    except Exception as e:
        logger.error(f"Data signature creation failed: {e}")
        return ""


def verify_data_signature(data: str, signature: str, secret_key: str) -> bool:
    """
    Verify HMAC signature for data integrity.
    
    Args:
        data: Original data
        signature: HMAC signature to verify
        secret_key: Secret key used for signing
    
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        expected_signature = create_data_signature(data, secret_key)
        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.error(f"Signature verification failed: {e}")
        return False
