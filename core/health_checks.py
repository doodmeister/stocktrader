"""
Health check system for the StockTrader dashboard.

This module provides comprehensive system health monitoring capabilities,
including directory structure validation, file integrity checks, resource
monitoring, and performance metrics.
"""

import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import streamlit as st

from utils.logger import get_logger
from utils.config import get_project_root

logger = get_logger(__name__)
project_root = get_project_root()


class HealthChecker:
    """
    Comprehensive health monitoring system for the dashboard.
    
    This class provides cached health checks with configurable thresholds
    and detailed system monitoring capabilities.
    """
    
    def __init__(self, cache_duration: int = 30):
        """
        Initialize the health checker.
        
        Args:
            cache_duration: Cache duration in seconds (default: 30)
        """
        self.cache_duration = cache_duration
        self.logger = get_logger(f"{__name__}.HealthChecker")
        
        # Health check configuration
        self.critical_dirs = {
            "Data Directory": "data",
            "Models Directory": "models", 
            "Logs Directory": "logs",
            "Dashboard Pages": "dashboard_pages",
            "Utils Module": "utils",
            "Core Module": "core",
            "Patterns Module": "patterns"
        }
        
        self.config_files = {
            "Environment Config": ".env",
            "Requirements": "requirements.txt",
            "Project Plan": "project_plan.md",
            "README": "readme.md"
        }
        
        # Performance thresholds
        self.min_disk_space_gb = 1  # Minimum free disk space in GB
        self.page_availability_threshold = 0.7  # 70% of pages should be active
    
    def perform_health_checks(self, pages_config: Optional[List] = None, 
                             state_manager=None) -> Dict[str, bool]:
        """
        Perform comprehensive system health checks with caching.
        
        Args:
            pages_config: List of page configurations to check
            state_manager: State manager instance to validate
            
        Returns:
            Dictionary of health check results
        """
        # Cache health checks for performance
        current_time = time.time()
        cache_key = "health_checks_cache"
        cache_time_key = "health_checks_timestamp"
        
        if (cache_key in st.session_state and 
            cache_time_key in st.session_state and
            current_time - st.session_state[cache_time_key] < self.cache_duration):
            return st.session_state[cache_key]
        
        checks = {}
        
        try:
            # Check critical directories
            self._check_directories(checks)
            
            # Check configuration files
            self._check_configuration_files(checks)
            
            # Check page availability
            self._check_page_availability(checks, pages_config)
            
            # Check system components
            self._check_system_components(checks, state_manager)
            
            # Check session state health
            self._check_session_state(checks)
            
            # Check disk space
            self._check_disk_space(checks)
            
            # Update cache
            st.session_state[cache_key] = checks
            st.session_state[cache_time_key] = current_time
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            checks["Health Check System"] = False
        
        return checks
    
    def _check_directories(self, checks: Dict[str, bool]) -> None:
        """Check existence of critical directories."""
        for check_name, dir_name in self.critical_dirs.items():
            directory_path = project_root / dir_name
            checks[check_name] = directory_path.exists() and directory_path.is_dir()
    
    def _check_configuration_files(self, checks: Dict[str, bool]) -> None:
        """Check existence of important configuration files."""
        for check_name, file_name in self.config_files.items():
            file_path = project_root / file_name
            checks[check_name] = file_path.exists() and file_path.is_file()
    
    def _check_page_availability(self, checks: Dict[str, bool], 
                               pages_config: Optional[List]) -> None:
        """Check dashboard page availability and health."""
        if pages_config:
            active_pages = len([p for p in pages_config if getattr(p, 'is_active', True)])
            total_pages = len(pages_config)
            
            if total_pages > 0:
                availability_ratio = active_pages / total_pages
                checks["Page Availability"] = availability_ratio >= self.page_availability_threshold
            else:
                checks["Page Availability"] = False
        else:
            checks["Page Availability"] = False
    
    def _check_system_components(self, checks: Dict[str, bool], state_manager) -> None:
        """Check critical system components."""
        # Check state manager
        checks["State Manager"] = state_manager is not None
          # Check logging system
        checks["Logging System"] = self.logger is not None
    
    def _check_session_state(self, checks: Dict[str, bool]) -> None:
        """Check session state health and required keys."""
        required_session_keys = ['current_page', 'page_history', 'dashboard_initialized']
        checks["Session State"] = all(key in st.session_state for key in required_session_keys)
    
    def _check_disk_space(self, checks: Dict[str, bool]) -> None:
        """Check available disk space."""
        try:
            total, used, free = shutil.disk_usage(project_root)
            free_gb = free // (2**30)  # Convert to GB
            checks["Disk Space"] = free_gb >= self.min_disk_space_gb
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
            checks["Disk Space"] = True  # Assume OK if can't check
    
    def render_health_status(self, pages_config: Optional[List] = None, 
                           state_manager=None) -> None:
        """Render health status in the sidebar."""
        # Get health checks (will use cache if available)
        health_checks = self.perform_health_checks(pages_config, state_manager)
        
        # Only show failing health checks to keep sidebar clean
        failing_checks = [name for name, status in health_checks.items() if not status]
        
        if failing_checks:
            st.subheader("⚠️ Health Issues")
            for check_name in failing_checks:
                st.error(f"❌ {check_name}")
        else:
            # All systems healthy - show compact status
            st.success("✅ All Systems Healthy")
            if st.button("🔍 View Details", help="Show all health check details"):
                with st.expander("Health Check Details", expanded=True):
                    for check_name, status in health_checks.items():
                        st.success(f"✅ {check_name}")
    
    def get_health_summary(self, pages_config: Optional[List] = None, 
                          state_manager=None) -> Dict[str, any]:
        """
        Get a comprehensive health summary for monitoring.
        
        Args:
            pages_config: List of page configurations
            state_manager: State manager instance
            
        Returns:
            Dictionary containing health summary information
        """
        health_checks = self.perform_health_checks(pages_config, state_manager)
        
        passing_checks = sum(1 for status in health_checks.values() if status)
        total_checks = len(health_checks)
        
        return {
            "overall_health": "healthy" if all(health_checks.values()) else "unhealthy",
            "passing_checks": passing_checks,
            "total_checks": total_checks,
            "health_percentage": (passing_checks / total_checks * 100) if total_checks > 0 else 0,
            "failing_checks": [name for name, status in health_checks.items() if not status],
            "timestamp": time.time(),
            "details": health_checks
        }
    
    def get_performance_metrics(self) -> Dict[str, any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {}
        
        # Session uptime
        if 'load_time' not in st.session_state:
            st.session_state.load_time = time.time()
        
        uptime = time.time() - st.session_state.load_time
        metrics["session_uptime_seconds"] = uptime
        
        # Cache status
        cache_key = "health_checks_cache"
        cache_time_key = "health_checks_timestamp"
        
        if cache_key in st.session_state and cache_time_key in st.session_state:
            cache_age = time.time() - st.session_state[cache_time_key]
            metrics["cache_age_seconds"] = cache_age
            metrics["cache_valid"] = cache_age < self.cache_duration
        else:
            metrics["cache_age_seconds"] = None
            metrics["cache_valid"] = False
        
        # Session state metrics
        metrics["session_state_keys"] = len(st.session_state.keys())
        
        # Navigation metrics
        metrics["navigation_count"] = st.session_state.get("navigation_count", 0)
        metrics["current_page"] = st.session_state.get("current_page", "unknown")
        
        return metrics
    
    def reset_cache(self) -> None:
        """Reset the health check cache to force fresh checks."""
        cache_keys = ["health_checks_cache", "health_checks_timestamp"]
        for key in cache_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        self.logger.info("Health check cache reset")
    
    def set_configuration(self, **kwargs) -> None:
        """
        Update health checker configuration.
        
        Keyword Args:
            cache_duration: Cache duration in seconds
            min_disk_space_gb: Minimum free disk space in GB
            page_availability_threshold: Minimum page availability ratio
        """
        if "cache_duration" in kwargs:
            self.cache_duration = kwargs["cache_duration"]
        
        if "min_disk_space_gb" in kwargs:
            self.min_disk_space_gb = kwargs["min_disk_space_gb"]
        
        if "page_availability_threshold" in kwargs:
            self.page_availability_threshold = kwargs["page_availability_threshold"]
        
        # Reset cache when configuration changes
        self.reset_cache()
        
        self.logger.info(f"Health checker configuration updated: {kwargs}")


# Convenience functions for backward compatibility
def perform_health_checks(pages_config: Optional[List] = None, 
                         state_manager=None) -> Dict[str, bool]:
    """
    Convenience function to perform health checks.
    
    Args:
        pages_config: List of page configurations
        state_manager: State manager instance
        
    Returns:
        Dictionary of health check results
    """
    checker = HealthChecker()
    return checker.perform_health_checks(pages_config, state_manager)


def render_health_status() -> None:
    """Convenience function to render health status in sidebar."""
    checker = HealthChecker()
    checker.render_health_status()


def get_health_summary(pages_config: Optional[List] = None, 
                      state_manager=None) -> Dict[str, any]:
    """
    Convenience function to get health summary.
    
    Args:
        pages_config: List of page configurations
        state_manager: State manager instance
        
    Returns:
        Dictionary containing health summary
    """
    checker = HealthChecker()
    return checker.get_health_summary(pages_config, state_manager)
