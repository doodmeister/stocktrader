"""
Authorization Module for StockTrader Security Package

Handles access control, permissions, and resource authorization.
Provides framework for role-based access control and resource permissions.
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Available permissions in the system."""
    READ_DASHBOARD = "read_dashboard"
    WRITE_DASHBOARD = "write_dashboard"
    EXECUTE_TRADES = "execute_trades"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_MODELS = "manage_models"
    ACCESS_SANDBOX = "access_sandbox"
    ACCESS_LIVE = "access_live"
    ADMIN_ACCESS = "admin_access"
    EXPORT_DATA = "export_data"
    IMPORT_DATA = "import_data"


class Role(Enum):
    """Available roles in the system."""
    VIEWER = "viewer"
    TRADER = "trader"
    ANALYST = "analyst"
    ADMIN = "admin"
    GUEST = "guest"


@dataclass
class UserContext:
    """User context for authorization decisions."""
    user_id: Optional[str] = None
    role: Role = Role.GUEST
    permissions: Set[Permission] = None
    session_valid: bool = False
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = set()


# Role-to-permissions mapping
ROLE_PERMISSIONS = {
    Role.GUEST: {
        Permission.READ_DASHBOARD,
        Permission.ACCESS_SANDBOX
    },
    Role.VIEWER: {
        Permission.READ_DASHBOARD,
        Permission.VIEW_ANALYTICS,
        Permission.ACCESS_SANDBOX,
        Permission.EXPORT_DATA
    },
    Role.TRADER: {
        Permission.READ_DASHBOARD,
        Permission.WRITE_DASHBOARD,
        Permission.EXECUTE_TRADES,
        Permission.VIEW_ANALYTICS,
        Permission.ACCESS_SANDBOX,
        Permission.ACCESS_LIVE,
        Permission.EXPORT_DATA,
        Permission.IMPORT_DATA
    },
    Role.ANALYST: {
        Permission.READ_DASHBOARD,
        Permission.WRITE_DASHBOARD,
        Permission.VIEW_ANALYTICS,
        Permission.MANAGE_MODELS,
        Permission.ACCESS_SANDBOX,
        Permission.EXPORT_DATA,
        Permission.IMPORT_DATA
    },
    Role.ADMIN: set(Permission)  # All permissions
}


def get_user_context() -> UserContext:
    """
    Get the current user context from session state.
    
    Returns:
        UserContext object with current user information
    """
    if not hasattr(st, 'session_state'):
        return UserContext()
    
    # Get user info from session state
    user_id = getattr(st.session_state, 'user_id', None)
    role_str = getattr(st.session_state, 'user_role', Role.GUEST.value)
    session_valid = getattr(st.session_state, 'security_initialized', False)
    
    # Convert role string to enum
    try:
        role = Role(role_str)
    except ValueError:
        role = Role.GUEST
    
    # Get permissions for the role
    permissions = ROLE_PERMISSIONS.get(role, set())
    
    return UserContext(
        user_id=user_id,
        role=role,
        permissions=permissions,
        session_valid=session_valid
    )


def check_access_permission(required_permission: Permission) -> bool:
    """
    Check if the current user has the required permission.
    
    Args:
        required_permission: Permission to check
        
    Returns:
        bool: True if user has permission, False otherwise
    """
    try:
        user_context = get_user_context()
        
        # Check if session is valid
        if not user_context.session_valid:
            logger.warning(f"Permission check failed: Invalid session for {required_permission}")
            return False
        
        # Check if user has the permission
        has_permission = required_permission in user_context.permissions
        
        if not has_permission:
            logger.warning(
                f"Permission denied: User {user_context.user_id} "
                f"(role: {user_context.role}) lacks {required_permission}"
            )
        
        return has_permission
        
    except Exception as e:
        logger.error(f"Error checking permission {required_permission}: {e}")
        return False


def validate_resource_access(resource_type: str, resource_id: str = None) -> bool:
    """
    Validate access to a specific resource.
    
    Args:
        resource_type: Type of resource (e.g., 'model', 'data', 'trade')
        resource_id: Optional specific resource identifier
        
    Returns:
        bool: True if access is allowed, False otherwise
    """
    try:
        user_context = get_user_context()
        
        # Check basic session validity
        if not user_context.session_valid:
            return False
        
        # Resource-specific access rules
        resource_permissions = {
            'dashboard': Permission.READ_DASHBOARD,
            'trading': Permission.EXECUTE_TRADES,
            'analytics': Permission.VIEW_ANALYTICS,
            'models': Permission.MANAGE_MODELS,
            'data_export': Permission.EXPORT_DATA,
            'data_import': Permission.IMPORT_DATA,
            'admin': Permission.ADMIN_ACCESS
        }
        
        required_permission = resource_permissions.get(resource_type)
        if required_permission:
            return check_access_permission(required_permission)
        
        # Default to deny access for unknown resources
        logger.warning(f"Unknown resource type: {resource_type}")
        return False
        
    except Exception as e:
        logger.error(f"Error validating resource access: {e}")
        return False


def get_user_permissions() -> Set[Permission]:
    """
    Get all permissions for the current user.
    
    Returns:
        Set of permissions for the current user
    """
    user_context = get_user_context()
    return user_context.permissions


def set_user_role(role: Role) -> bool:
    """
    Set the user role in session state.
    
    Args:
        role: Role to assign to the user
        
    Returns:
        bool: True if role was set successfully
    """
    try:
        if not hasattr(st, 'session_state'):
            return False
        
        st.session_state.user_role = role.value
        logger.info(f"User role set to: {role.value}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting user role: {e}")
        return False


def require_permission(permission: Permission):
    """
    Decorator to require a specific permission for a function.
    
    Args:
        permission: Required permission
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not check_access_permission(permission):
                st.error(f"Access denied: {permission.value} permission required")
                st.stop()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(role: Role):
    """
    Decorator to require a specific role for a function.
    
    Args:
        role: Required role
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            user_context = get_user_context()
            if user_context.role != role and user_context.role != Role.ADMIN:
                st.error(f"Access denied: {role.value} role required")
                st.stop()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def check_trading_environment_access(use_live: bool = False) -> bool:
    """
    Check if user has access to trading environment.
    
    Args:
        use_live: Whether this is for live trading
        
    Returns:
        bool: True if access is allowed
    """
    if use_live:
        return check_access_permission(Permission.ACCESS_LIVE)
    else:
        return check_access_permission(Permission.ACCESS_SANDBOX)


def audit_access_attempt(resource: str, granted: bool, user_context: UserContext = None):
    """
    Audit an access attempt for security monitoring.
    
    Args:
        resource: Resource that was accessed
        granted: Whether access was granted
        user_context: Optional user context
    """
    if user_context is None:
        user_context = get_user_context()
    
    log_data = {
        'resource': resource,
        'granted': granted,
        'user_id': user_context.user_id,
        'role': user_context.role.value,
        'timestamp': st.session_state.get('last_activity', 'unknown')
    }
    
    if granted:
        logger.info(f"Access granted: {log_data}")
    else:
        logger.warning(f"Access denied: {log_data}")


def get_accessible_features() -> List[str]:
    """
    Get list of features accessible to the current user.
    
    Returns:
        List of accessible feature names
    """
    user_context = get_user_context()
    features = []
    
    if Permission.READ_DASHBOARD in user_context.permissions:
        features.append("dashboard")
    
    if Permission.EXECUTE_TRADES in user_context.permissions:
        features.append("trading")
    
    if Permission.VIEW_ANALYTICS in user_context.permissions:
        features.append("analytics")
    
    if Permission.MANAGE_MODELS in user_context.permissions:
        features.append("models")
    
    if Permission.EXPORT_DATA in user_context.permissions:
        features.append("data_export")
    
    if Permission.ADMIN_ACCESS in user_context.permissions:
        features.append("admin")
    
    return features
