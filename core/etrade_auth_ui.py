"""
E*Trade Authentication UI Component

Provides Streamlit UI components for E*Trade OAuth authentication flow
following the official examples pattern with integrated security framework.
"""

import streamlit as st
import os
from typing import Optional, Dict, Any
from core.etrade_client import ETradeClient, ETradeAuthenticationError, ETradeAPIError
from security.etrade_security import SecureETradeManager
from security.authorization import check_etrade_access, validate_etrade_environment_access
from utils.logger import get_dashboard_logger

logger = get_dashboard_logger(__name__)


def render_etrade_authentication() -> Optional[ETradeClient]:
    """
    Render E*Trade authentication UI with integrated security framework.
    
    Returns:
        ETradeClient instance if authenticated, None otherwise
    """
    st.sidebar.header("🔑 E*Trade Authentication")
    
    # Initialize security framework
    if not SecureETradeManager.initialize_security():
        st.sidebar.error("❌ Security initialization failed")
        return None
    
    # Check if already authenticated
    client = SecureETradeManager.get_authenticated_client()
    if client:
        st.sidebar.success("✅ Authenticated with E*Trade")
        
        # Show account info
        if hasattr(client, 'accounts') and client.accounts:
            st.sidebar.subheader("Available Accounts")
            for i, account in enumerate(client.accounts):
                account_desc = account.get('accountDesc', 'Unknown')
                account_id = account.get('accountId', 'Unknown')
                account_mode = account.get('accountMode', 'Unknown')
                st.sidebar.text(f"{i+1}. {account_desc} ({account_mode})")
                st.sidebar.text(f"   ID: {account_id}")
        
        # Show environment and permissions
        env_type = "SANDBOX" if client.sandbox else "LIVE"
        st.sidebar.info(f"🌐 Environment: {env_type}")
        
        # Show available operations based on permissions
        operations = []
        if check_etrade_access('market_data', not client.sandbox):
            operations.append("📊 Market Data")
        if check_etrade_access('orders', not client.sandbox):
            operations.append("📈 Trading")
        
        if operations:
            st.sidebar.subheader("Available Operations")
            for op in operations:
                st.sidebar.text(f"✓ {op}")
        
        # Logout button
        if st.sidebar.button("🔄 Logout", key="etrade_logout"):
            SecureETradeManager.logout()
            st.rerun()
        
        return client
    
    # Authentication flow
    _render_authentication_flow()
    return None


def _render_authentication_flow():
    """Render the E*Trade OAuth authentication flow."""
    
    # Initialize auth stage
    if 'etrade_auth_stage' not in st.session_state:
        st.session_state.etrade_auth_stage = 'credentials'
    
    if st.session_state.etrade_auth_stage == 'credentials':
        _render_credentials_input()
    elif st.session_state.etrade_auth_stage == 'authorization':
        _render_authorization_step()
    elif st.session_state.etrade_auth_stage == 'verification':
        _render_verification_step()


def _render_credentials_input():
    """Render credential input form with security validation."""
    st.sidebar.subheader("Step 1: Enter Credentials")
    
    # Check basic E*Trade connection permission
    if not check_etrade_access('connect'):
        st.sidebar.error("❌ Access denied: E*Trade connection permission required")
        return
    
    # Load from environment as defaults
    consumer_key = st.sidebar.text_input(
        "Consumer Key",
        value=os.getenv('ETRADE_CONSUMER_KEY', ''),
        type="password",
        help="Your E*Trade Consumer Key"
    )
    
    consumer_secret = st.sidebar.text_input(
        "Consumer Secret", 
        value=os.getenv('ETRADE_CONSUMER_SECRET', ''),
        type="password",
        help="Your E*Trade Consumer Secret"
    )
    
    # Environment selection with permission checking
    environment_options = ["Sandbox"]
    if check_etrade_access('connect', use_live=True):
        environment_options.append("Live")
    
    if len(environment_options) == 1:
        st.sidebar.info("ℹ️ Only sandbox access available with current permissions")
    
    environment = st.sidebar.radio(
        "Environment",
        environment_options,
        index=0,
        help="Sandbox for testing, Live for real trading"
    )
    
    use_sandbox = (environment == "Sandbox")
    
    # Live trading warning and validation
    if not use_sandbox:
        if not validate_etrade_environment_access(use_live=True):
            st.sidebar.error("❌ Access denied: Insufficient permissions for live trading")
            return
        
        st.sidebar.warning("⚠️ LIVE TRADING MODE - Real money will be used!")
        confirm_live = st.sidebar.checkbox(
            "I understand this is live trading with real money",
            key="confirm_live_trading"
        )
        if not confirm_live:
            st.sidebar.error("Please confirm you understand live trading risks")
            return
    
    # Connect button
    if st.sidebar.button("🔗 Connect to E*Trade", key="etrade_connect"):
        if not consumer_key or not consumer_secret:
            st.sidebar.error("Please enter both Consumer Key and Consumer Secret")
            return
        
        try:
            # Store credentials securely
            if SecureETradeManager.store_credentials(consumer_key, consumer_secret, use_sandbox):
                st.session_state.etrade_auth_stage = 'authorization'
                st.rerun()
            else:
                st.sidebar.error("Failed to store credentials securely")
            
        except Exception as e:
            st.sidebar.error(f"Failed to initialize client: {e}")
            logger.error(f"Failed to initialize E*Trade client: {e}")


def _render_authorization_step():
    """Render authorization step with browser instructions."""
    st.sidebar.subheader("Step 2: Browser Authorization")
    
    # Get stored credentials
    if SecureETradeManager.ENCRYPTED_CREDENTIALS not in st.session_state:
        st.sidebar.error("Credentials not found")
        st.session_state.etrade_auth_stage = 'credentials'
        st.rerun()
        return
    
    try:
        # Get temporary credentials for OAuth flow
        consumer_key = st.session_state.get('_temp_etrade_key')
        consumer_secret = st.session_state.get('_temp_etrade_secret')
        
        if not consumer_key or not consumer_secret:
            st.sidebar.error("Temporary credentials not available")
            st.session_state.etrade_auth_stage = 'credentials'
            st.rerun()
            return
        
        # Get environment setting
        creds = st.session_state[SecureETradeManager.ENCRYPTED_CREDENTIALS]
        use_sandbox = creds.get('use_sandbox', True)
        
        # Create temporary client for OAuth flow
        client = ETradeClient(consumer_key, consumer_secret, use_sandbox)
        
        # Get authorization URL
        request_token, request_token_secret = client.etrade.get_request_token(
            params={"oauth_callback": "oob", "format": "json"}
        )
        
        authorize_url = client.etrade.authorize_url.format(
            client.etrade.consumer_key, request_token
        )
        
        # Store tokens for verification step
        st.session_state.etrade_request_token = request_token
        st.session_state.etrade_request_token_secret = request_token_secret
        st.session_state.etrade_temp_client = client
        
        # Instructions
        st.sidebar.markdown("""
        **Instructions:**
        1. Click the link below to open E*Trade authorization
        2. Log in to your E*Trade account
        3. Accept the application authorization
        4. Copy the verification code shown
        5. Return here and enter the code
        """)
        
        # Authorization link
        st.sidebar.markdown(f"[🌐 Authorize Application]({authorize_url})")
        
        # Manual URL display
        with st.sidebar.expander("Manual URL"):
            st.code(authorize_url)
        
        # Next step button
        if st.sidebar.button("✅ I've authorized the app", key="auth_complete"):
            st.session_state.etrade_auth_stage = 'verification'
            st.rerun()
        
        # Back button
        if st.sidebar.button("🔙 Back", key="auth_back"):
            st.session_state.etrade_auth_stage = 'credentials'
            _cleanup_temp_auth_data()
            st.rerun()
            
    except Exception as e:
        st.sidebar.error(f"Authorization failed: {e}")
        logger.error(f"E*Trade authorization failed: {e}")


def _render_verification_step():
    """Render verification code input step with secure authentication."""
    st.sidebar.subheader("Step 3: Enter Verification Code")
    
    request_token = st.session_state.get('etrade_request_token')
    request_token_secret = st.session_state.get('etrade_request_token_secret')
    
    if not all([request_token, request_token_secret]):
        st.sidebar.error("Missing authentication data")
        st.session_state.etrade_auth_stage = 'credentials'
        _cleanup_temp_auth_data()
        st.rerun()
        return
    
    # Verification code input
    verification_code = st.sidebar.text_input(
        "Verification Code",
        help="Enter the verification code from E*Trade",
        placeholder="Enter code here..."
    )
    
    # Complete authentication
    if st.sidebar.button("🔐 Complete Authentication", key="complete_auth"):
        if not verification_code:
            st.sidebar.error("Please enter the verification code")
            return
        
        try:
            # Use secure authentication flow
            client = SecureETradeManager.create_authenticated_client(verification_code)
            
            if client:
                st.session_state.etrade_auth_stage = 'credentials'  # Reset for next time
                _cleanup_temp_auth_data()
                st.sidebar.success("🎉 Authentication successful!")
                st.rerun()
            else:
                st.sidebar.error("Authentication failed - please try again")
            
        except ETradeAuthenticationError as e:
            st.sidebar.error(f"Authentication failed: {e}")
            logger.error(f"E*Trade authentication failed: {e}")
        except ETradeAPIError as e:
            st.sidebar.error(f"API error: {e}")
            logger.error(f"E*Trade API error: {e}")
        except Exception as e:
            st.sidebar.error(f"Unexpected error: {e}")
            logger.error(f"Unexpected E*Trade error: {e}")
    
    # Back button
    if st.sidebar.button("🔙 Back to Authorization", key="verify_back"):
        st.session_state.etrade_auth_stage = 'authorization'
        st.rerun()


def _cleanup_temp_auth_data():
    """Clean up temporary authentication data."""
    temp_keys = [
        'etrade_client_temp',
        'etrade_temp_client',
        'etrade_request_token', 
        'etrade_request_token_secret'
    ]
    
    for key in temp_keys:
        if key in st.session_state:
            del st.session_state[key]


def get_etrade_client() -> Optional[ETradeClient]:
    """
    Get the authenticated E*Trade client with security validation.
    
    Returns:
        ETradeClient instance if authenticated, None otherwise
    """
    return SecureETradeManager.get_authenticated_client()


def is_etrade_authenticated() -> bool:
    """
    Check if E*Trade client is authenticated with security validation.
    
    Returns:
        True if authenticated, False otherwise
    """
    client = get_etrade_client()
    return client is not None and client.session is not None


def validate_etrade_operation(operation: str) -> bool:
    """
    Validate if current user can perform E*Trade operation.
    
    Args:
        operation: Operation to validate
        
    Returns:
        bool: True if operation is allowed
    """
    return SecureETradeManager.validate_operation_access(operation)


def get_etrade_session_info() -> Dict[str, Any]:
    """
    Get E*Trade session information for monitoring.
    
    Returns:
        Dictionary with session information
    """
    return SecureETradeManager.get_session_info()
