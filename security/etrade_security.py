"""
E*Trade Secure Credential Manager

Integrates E*Trade authentication with the StockTrader security framework.
Provides secure credential storage, session management, and audit logging.
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any, Tuple
import time
from datetime import datetime

from security.encryption import (
    create_secure_token, 
    hash_password, 
    verify_password,
    validate_session_token
)
from security.authentication import validate_session_security
from security.authorization import (
    check_etrade_access,
    validate_etrade_environment_access,
    audit_access_attempt
)
from core.etrade_client import ETradeClient, ETradeAuthenticationError

logger = logging.getLogger(__name__)


class SecureETradeManager:
    """
    Secure E*Trade credential and session manager.
    
    Integrates E*Trade authentication with the StockTrader security framework
    to provide secure credential storage, session validation, and audit logging.
    """
    
    # Session state keys
    ENCRYPTED_CREDENTIALS = "etrade_encrypted_creds"
    CLIENT_INSTANCE = "etrade_secure_client"
    AUTH_TOKEN = "etrade_auth_token"
    LAST_ACTIVITY = "etrade_last_activity"
    AUTH_TIMESTAMP = "etrade_auth_time"
    SESSION_SALT = "etrade_session_salt"
    
    # Security settings
    SESSION_TIMEOUT = 4 * 60 * 60  # 4 hours
    CREDENTIAL_HASH_ITERATIONS = 100000
    
    @classmethod
    def initialize_security(cls) -> bool:
        """
        Initialize secure E*Trade session state.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Validate main session security first
            if not validate_session_security():
                logger.error("Main session security validation failed")
                return False
            
            # Initialize E*Trade specific security state
            if cls.SESSION_SALT not in st.session_state:
                st.session_state[cls.SESSION_SALT] = create_secure_token(32)
            
            if cls.AUTH_TOKEN not in st.session_state:
                st.session_state[cls.AUTH_TOKEN] = None
            
            if cls.LAST_ACTIVITY not in st.session_state:
                st.session_state[cls.LAST_ACTIVITY] = time.time()
            
            logger.info("E*Trade security initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize E*Trade security: {e}")
            return False
    
    @classmethod
    def store_credentials(cls, consumer_key: str, consumer_secret: str, 
                         use_sandbox: bool = True) -> bool:
        """
        Securely store E*Trade credentials.
        
        Args:
            consumer_key: E*Trade consumer key
            consumer_secret: E*Trade consumer secret
            use_sandbox: Whether to use sandbox environment
            
        Returns:
            bool: True if credentials stored successfully
        """
        try:
            if not cls.initialize_security():
                return False
            
            # Create credential hash for validation
            salt = st.session_state[cls.SESSION_SALT]
            
            # Hash credentials with salt
            key_hash, _ = hash_password(consumer_key, salt)
            secret_hash, _ = hash_password(consumer_secret, salt)
            
            # Store encrypted credentials
            encrypted_creds = {
                'consumer_key_hash': key_hash,
                'consumer_secret_hash': secret_hash,
                'use_sandbox': use_sandbox,
                'created_at': time.time(),
                'salt': salt
            }
            
            st.session_state[cls.ENCRYPTED_CREDENTIALS] = encrypted_creds
            st.session_state[cls.LAST_ACTIVITY] = time.time()
            
            # Store raw credentials temporarily for authentication
            # (Will be cleared after authentication)
            st.session_state['_temp_etrade_key'] = consumer_key
            st.session_state['_temp_etrade_secret'] = consumer_secret
            
            logger.info("E*Trade credentials stored securely")
            audit_access_attempt("etrade_credential_store", True)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store E*Trade credentials: {e}")
            audit_access_attempt("etrade_credential_store", False)
            return False
    
    @classmethod
    def validate_credentials(cls, consumer_key: str, consumer_secret: str) -> bool:
        """
        Validate stored credentials against provided ones.
        
        Args:
            consumer_key: Consumer key to validate
            consumer_secret: Consumer secret to validate
            
        Returns:
            bool: True if credentials match stored ones
        """
        try:
            if cls.ENCRYPTED_CREDENTIALS not in st.session_state:
                return False
            
            creds = st.session_state[cls.ENCRYPTED_CREDENTIALS]
            salt = creds.get('salt', st.session_state.get(cls.SESSION_SALT))
            
            # Verify credentials against stored hashes
            key_valid = verify_password(
                consumer_key, 
                creds['consumer_key_hash'], 
                salt
            )
            secret_valid = verify_password(
                consumer_secret, 
                creds['consumer_secret_hash'], 
                salt
            )
            
            return key_valid and secret_valid
            
        except Exception as e:
            logger.error(f"Failed to validate E*Trade credentials: {e}")
            return False
    
    @classmethod
    def create_authenticated_client(cls, verification_code: str) -> Optional[ETradeClient]:
        """
        Create an authenticated E*Trade client with security validation.
        
        Args:
            verification_code: OAuth verification code
            
        Returns:
            ETradeClient instance if successful, None otherwise
        """
        try:
            # Validate session security
            if not validate_session_security():
                logger.error("Session security validation failed for E*Trade client creation")
                return None
            
            # Check credentials exist
            if cls.ENCRYPTED_CREDENTIALS not in st.session_state:
                logger.error("No stored E*Trade credentials found")
                return None
            
            # Get temporary credentials
            consumer_key = st.session_state.get('_temp_etrade_key')
            consumer_secret = st.session_state.get('_temp_etrade_secret')
            
            if not consumer_key or not consumer_secret:
                logger.error("Temporary credentials not available")
                return None
            
            # Get environment setting
            creds = st.session_state[cls.ENCRYPTED_CREDENTIALS]
            use_sandbox = creds.get('use_sandbox', True)
            
            # Validate environment access
            if not validate_etrade_environment_access(not use_sandbox):
                logger.error(f"Access denied for E*Trade environment (live={not use_sandbox})")
                return None
            
            # Create and authenticate client
            client = ETradeClient(consumer_key, consumer_secret, use_sandbox)
            
            if client.authenticate(verification_code):
                # Store authenticated client
                st.session_state[cls.CLIENT_INSTANCE] = client
                st.session_state[cls.AUTH_TOKEN] = create_secure_token()
                st.session_state[cls.AUTH_TIMESTAMP] = time.time()
                st.session_state[cls.LAST_ACTIVITY] = time.time()
                
                # Clear temporary credentials
                cls._clear_temp_credentials()
                
                logger.info("E*Trade client authenticated successfully")
                audit_access_attempt("etrade_authentication", True)
                return client
            else:
                logger.error("E*Trade authentication failed")
                audit_access_attempt("etrade_authentication", False)
                return None
                
        except ETradeAuthenticationError as e:
            logger.error(f"E*Trade authentication error: {e}")
            audit_access_attempt("etrade_authentication", False)
            return None
        except Exception as e:
            logger.error(f"Failed to create authenticated E*Trade client: {e}")
            audit_access_attempt("etrade_authentication", False)
            return None
    
    @classmethod
    def get_authenticated_client(cls) -> Optional[ETradeClient]:
        """
        Get the current authenticated E*Trade client with session validation.
        
        Returns:
            ETradeClient instance if valid session, None otherwise
        """
        try:
            # Validate main session
            if not validate_session_security():
                cls._clear_authentication()
                return None
            
            # Check if client exists
            if cls.CLIENT_INSTANCE not in st.session_state:
                return None
            
            # Check session timeout
            if not cls._check_auth_timeout():
                cls._clear_authentication()
                return None
            
            # Update activity timestamp
            st.session_state[cls.LAST_ACTIVITY] = time.time()
            
            client = st.session_state[cls.CLIENT_INSTANCE]
            
            # Validate client session
            if not hasattr(client, 'session') or not client.session:
                cls._clear_authentication()
                return None
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to get authenticated E*Trade client: {e}")
            cls._clear_authentication()
            return None
    
    @classmethod
    def validate_operation_access(cls, operation: str) -> bool:
        """
        Validate access for specific E*Trade operations.
        
        Args:
            operation: Operation to validate ('market_data', 'orders', etc.)
            
        Returns:
            bool: True if access is allowed
        """
        try:
            client = cls.get_authenticated_client()
            if not client:
                return False
            
            # Check operation-specific permissions
            use_live = not client.sandbox
            return check_etrade_access(operation, use_live)
            
        except Exception as e:
            logger.error(f"Failed to validate E*Trade operation access: {e}")
            return False
    
    @classmethod
    def _check_auth_timeout(cls) -> bool:
        """Check if authentication session has timed out."""
        if cls.LAST_ACTIVITY not in st.session_state:
            return False
        
        time_since_activity = time.time() - st.session_state[cls.LAST_ACTIVITY]
        return time_since_activity < cls.SESSION_TIMEOUT
    
    @classmethod
    def _clear_temp_credentials(cls) -> None:
        """Clear temporary credentials from session state."""
        temp_keys = ['_temp_etrade_key', '_temp_etrade_secret']
        for key in temp_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    @classmethod
    def _clear_authentication(cls) -> None:
        """Clear all E*Trade authentication data."""
        auth_keys = [
            cls.CLIENT_INSTANCE,
            cls.AUTH_TOKEN,
            cls.AUTH_TIMESTAMP,
            cls.ENCRYPTED_CREDENTIALS
        ]
        
        for key in auth_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        cls._clear_temp_credentials()
        logger.info("E*Trade authentication cleared")
    
    @classmethod
    def logout(cls) -> None:
        """Logout and clear all E*Trade session data."""
        cls._clear_authentication()
        audit_access_attempt("etrade_logout", True)
    
    @classmethod
    def get_session_info(cls) -> Dict[str, Any]:
        """
        Get E*Trade session information for monitoring.
        
        Returns:
            Dictionary with session information
        """
        return {
            'authenticated': cls.CLIENT_INSTANCE in st.session_state,
            'has_credentials': cls.ENCRYPTED_CREDENTIALS in st.session_state,
            'last_activity': st.session_state.get(cls.LAST_ACTIVITY),
            'auth_timestamp': st.session_state.get(cls.AUTH_TIMESTAMP),
            'session_valid': cls._check_auth_timeout(),
            'environment': 'sandbox' if st.session_state.get(cls.ENCRYPTED_CREDENTIALS, {}).get('use_sandbox', True) else 'live'
        }
