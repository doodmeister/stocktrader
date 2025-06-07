"""
Example Dashboard Using the Streamlit Dashboard Template

This file demonstrates how to create a custom dashboard using the comprehensive
template. It shows the basic customization steps and implementation patterns.

This example creates a simple analytics dashboard that demonstrates:
- Custom configuration
- Data processing and visualization
- Interactive controls
- Error handling
- Performance monitoring

To use this example:
1. Ensure the template file is available
2. Install required dependencies
3. Run with: streamlit run dashboard_pages/example_analytics_dashboard.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import the template (adjust path as needed)
try:
    # Add the templates directory to Python path
    template_dir = Path(__file__).parent.parent / "templates"
    sys.path.insert(0, str(template_dir))
    
    from streamlit_dashboard_template import (
        TemplateDashboard,
        TemplateConfig,
        TemplateValidator,
        handle_dashboard_errors,
        SESSION_STATE_KEYS
    )
except ImportError as e:
    st.error(f"Failed to import template: {e}")
    st.stop()

# =============================================================================
# CUSTOM CONFIGURATION
# =============================================================================

# Override page configuration for analytics dashboard
PAGE_CONFIG = {
    "page_title": "Analytics Dashboard",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Custom session state keys
ANALYTICS_SESSION_KEYS = {
    **SESSION_STATE_KEYS,  # Include base template keys
    "analytics_data": None,
    "selected_metrics": [],
    "date_range": None,
    "chart_type": "line",
    "analytics_filters": {}
}

# =============================================================================
# CUSTOM CONFIGURATION CLASS
# =============================================================================

class AnalyticsConfig(TemplateConfig):
    """Custom configuration for the analytics dashboard."""
    
    def __init__(self):
        super().__init__()
        
        # Add analytics-specific configuration
        self.config.update({
            # Analytics settings
            "default_chart_type": "line",
            "max_data_points": 10000,
            "chart_height": 400,
            "enable_real_time": False,
            
            # Data settings
            "data_sources": ["csv", "xlsx", "json", "api"],
            "default_metrics": ["revenue", "users", "sessions"],
            "aggregation_methods": ["sum", "mean", "count", "max", "min"],
            
            # Display settings
            "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
            "chart_theme": "plotly",
            "show_data_table": True,
            
            # Performance settings
            "cache_charts": True,
            "chart_cache_ttl": 600,  # 10 minutes
        })

# =============================================================================
# CUSTOM VALIDATOR CLASS
# =============================================================================

class AnalyticsValidator(TemplateValidator):
    """Custom validation for analytics dashboard."""
    
    @staticmethod
    def validate_time_series_data(df: pd.DataFrame, date_column: str = "date") -> bool:
        """Validate time series data structure."""
        try:
            # Check if date column exists
            if date_column not in df.columns:
                raise ValueError(f"Date column '{date_column}' not found")
            
            # Check if date column can be parsed
            pd.to_datetime(df[date_column])
            
            # Check for reasonable date range
            dates = pd.to_datetime(df[date_column])
            date_range = dates.max() - dates.min()
            if date_range.days > 10000:  # More than ~27 years
                raise ValueError("Date range too large")
            
            return True
            
        except Exception as e:
            raise ValueError(f"Invalid time series data: {e}")
    
    @staticmethod
    def validate_metrics_selection(metrics: List[str], available_metrics: List[str]) -> bool:
        """Validate selected metrics against available options."""
        try:
            if not metrics:
                raise ValueError("No metrics selected")
            
            invalid_metrics = set(metrics) - set(available_metrics)
            if invalid_metrics:
                raise ValueError(f"Invalid metrics: {invalid_metrics}")
            
            if len(metrics) > 10:
                raise ValueError("Too many metrics selected (max 10)")
            
            return True
            
        except Exception as e:
            raise ValueError(f"Invalid metrics selection: {e}")

# =============================================================================
# MAIN ANALYTICS DASHBOARD CLASS
# =============================================================================

class AnalyticsDashboard(TemplateDashboard):
    """
    Analytics Dashboard using the template framework.
    
    This dashboard provides comprehensive analytics capabilities including:
    - Time series visualization
    - Interactive metric selection
    - Data filtering and aggregation
    - Export functionality
    - Real-time data updates
    """
    
    def __init__(self):
        """Initialize the analytics dashboard."""
        # Use custom configuration and validator
        self.config = AnalyticsConfig()
        self.validator = AnalyticsValidator()
        
        # Initialize session state with custom keys
        self._initialize_custom_session_state()
        
        # Generate sample data if no data is loaded
        if st.session_state.analytics_data is None:
            self._generate_sample_data()
    
    def _initialize_custom_session_state(self) -> None:
        """Initialize session state with analytics-specific keys."""
        for key, default_value in ANALYTICS_SESSION_KEYS.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Set default values for analytics
        if not st.session_state.selected_metrics:
            st.session_state.selected_metrics = self.config.get("default_metrics", ["revenue"])
        
        if st.session_state.date_range is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            st.session_state.date_range = (start_date, end_date)
    
    def _generate_sample_data(self) -> None:
        """Generate sample analytics data for demonstration."""
        try:
            # Generate 90 days of sample data
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=90),
                end=datetime.now(),
                freq='D'
            )
            
            # Generate realistic sample data
            np.random.seed(42)  # For reproducible results
            data = {
                'date': dates,
                'revenue': np.random.normal(10000, 2000, len(dates)).cumsum(),
                'users': np.random.poisson(500, len(dates)).cumsum(),
                'sessions': np.random.poisson(1000, len(dates)).cumsum(),
                'conversions': np.random.poisson(50, len(dates)).cumsum(),
                'bounce_rate': np.random.uniform(0.3, 0.7, len(dates)),
                'avg_session_duration': np.random.normal(180, 60, len(dates))
            }
            
            df = pd.DataFrame(data)
            
            # Validate the generated data
            self.validator.validate_time_series_data(df, 'date')
            
            # Store in session state
            st.session_state.analytics_data = df
            
        except Exception as e:
            st.error(f"Failed to generate sample data: {e}")
    
    @handle_dashboard_errors
    def render_header(self) -> None:
        """Render custom header for analytics dashboard."""
        st.markdown("""
        <div class="main-header">
            <h1>📈 Analytics Dashboard</h1>
            <p>Comprehensive analytics and data visualization platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add quick stats
        if st.session_state.analytics_data is not None:
            df = st.session_state.analytics_data
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_revenue = df['revenue'].iloc[-1]
                st.metric("Total Revenue", f"${total_revenue:,.0f}")
            
            with col2:
                total_users = df['users'].iloc[-1]
                st.metric("Total Users", f"{total_users:,}")
            
            with col3:
                total_sessions = df['sessions'].iloc[-1]
                st.metric("Total Sessions", f"{total_sessions:,}")
            
            with col4:
                avg_bounce_rate = df['bounce_rate'].mean()
                st.metric("Avg Bounce Rate", f"{avg_bounce_rate:.1%}")
    
    @handle_dashboard_errors
    def render_sidebar(self) -> None:
        """Render custom sidebar for analytics dashboard."""
        with st.sidebar:
            st.header("⚙️ Analytics Controls")
            
            # Data controls
            with st.expander("📊 Data Controls", expanded=True):
                # Metric selection
                if st.session_state.analytics_data is not None:
                    df = st.session_state.analytics_data
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    selected_metrics = st.multiselect(
                        "Select Metrics",
                        options=numeric_columns,
                        default=st.session_state.selected_metrics[:3],  # Limit to 3 for performance
                        help="Choose metrics to display in charts"
                    )
                    
                    if selected_metrics:
                        try:
                            self.validator.validate_metrics_selection(selected_metrics, numeric_columns)
                            st.session_state.selected_metrics = selected_metrics
                        except ValueError as e:
                            st.error(f"Invalid selection: {e}")
                
                # Date range selection
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=st.session_state.date_range[0],
                        help="Select start date for analysis"
                    )
                
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=st.session_state.date_range[1],
                        help="Select end date for analysis"
                    )
                
                st.session_state.date_range = (start_date, end_date)
                
                # Chart type selection
                chart_type = st.selectbox(
                    "Chart Type",
                    options=["line", "bar", "area", "scatter"],
                    index=0,
                    help="Select visualization type"
                )
                st.session_state.chart_type = chart_type
            
            # Display controls
            with st.expander("🎨 Display Options", expanded=False):
                show_data_table = st.checkbox(
                    "Show Data Table",
                    value=self.config.get("show_data_table", True),
                    help="Display raw data table below charts"
                )
                
                chart_height = st.slider(
                    "Chart Height",
                    min_value=300,
                    max_value=800,
                    value=self.config.get("chart_height", 400),
                    step=50,
                    help="Adjust chart height in pixels"
                )
                
                # Update configuration
                self.config.update({
                    "show_data_table": show_data_table,
                    "chart_height": chart_height
                })
            
            # Export controls
            with st.expander("📁 Data Export", expanded=False):
                if st.session_state.analytics_data is not None:
                    df = st.session_state.analytics_data
                    
                    # Filter data by date range
                    filtered_df = self._filter_data_by_date_range(df)
                    
                    # CSV download
                    csv_data = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"analytics_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # JSON download
                    json_data = filtered_df.to_json(orient='records', date_format='iso')
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"analytics_data_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
        
        # Call parent sidebar for template features
        super().render_sidebar()
    
    def _filter_data_by_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe by selected date range."""
        try:
            start_date, end_date = st.session_state.date_range
            
            # Convert to datetime if needed
            df_filtered = df.copy()
            df_filtered['date'] = pd.to_datetime(df_filtered['date'])
            
            # Filter by date range
            mask = (df_filtered['date'].dt.date >= start_date) & (df_filtered['date'].dt.date <= end_date)
            return df_filtered[mask]
            
        except Exception as e:
            st.error(f"Error filtering data: {e}")
            return df
    
    @handle_dashboard_errors
    def render_main_content(self) -> None:
        """Render the main analytics dashboard content."""
        st.header("📊 Analytics Overview")
        
        if st.session_state.analytics_data is None:
            st.warning("No data available. Please upload data or use sample data.")
            return
        
        # Get filtered data
        df = self._filter_data_by_date_range(st.session_state.analytics_data)
        
        if df.empty:
            st.warning("No data available for selected date range.")
            return
        
        # Render charts
        self._render_metrics_charts(df)
        
        # Render data table if enabled
        if self.config.get("show_data_table", True):
            self._render_data_table(df)
        
        # Render summary statistics
        self._render_summary_statistics(df)
    
    @handle_dashboard_errors
    def _render_metrics_charts(self, df: pd.DataFrame) -> None:
        """Render charts for selected metrics."""
        if not st.session_state.selected_metrics:
            st.info("Please select metrics to display charts.")
            return
        
        # Create charts for each selected metric
        for metric in st.session_state.selected_metrics:
            if metric in df.columns:
                st.subheader(f"📈 {metric.replace('_', ' ').title()}")
                
                # Create chart based on selected type
                chart_type = st.session_state.chart_type
                
                if chart_type == "line":
                    fig = px.line(df, x='date', y=metric, title=f"{metric} Over Time")
                elif chart_type == "bar":
                    fig = px.bar(df, x='date', y=metric, title=f"{metric} Over Time")
                elif chart_type == "area":
                    fig = px.area(df, x='date', y=metric, title=f"{metric} Over Time")
                elif chart_type == "scatter":
                    fig = px.scatter(df, x='date', y=metric, title=f"{metric} Over Time")
                else:
                    fig = px.line(df, x='date', y=metric, title=f"{metric} Over Time")
                
                # Update layout
                fig.update_layout(
                    height=self.config.get("chart_height", 400),
                    xaxis_title="Date",
                    yaxis_title=metric.replace('_', ' ').title()
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
    
    @handle_dashboard_errors
    def _render_data_table(self, df: pd.DataFrame) -> None:
        """Render data table with filtering options."""
        st.subheader("📋 Data Table")
        
        # Show basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", len(df))
        with col2:
            st.metric("Date Range", f"{len(df)} days")
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory", f"{memory_mb:.1f} MB")
        
        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            height=300
        )
    
    @handle_dashboard_errors
    def _render_summary_statistics(self, df: pd.DataFrame) -> None:
        """Render summary statistics for the data."""
        st.subheader("📊 Summary Statistics")
        
        # Calculate statistics for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            stats_df = numeric_df.describe()
            st.dataframe(stats_df, use_container_width=True)
            
            # Correlation matrix for selected metrics
            if len(st.session_state.selected_metrics) > 1:
                st.subheader("🔗 Correlation Matrix")
                
                # Get correlation matrix for selected metrics
                selected_data = df[st.session_state.selected_metrics]
                corr_matrix = selected_data.corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu"
                )
                
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the analytics dashboard."""
    try:
        # Create and run the analytics dashboard
        dashboard = AnalyticsDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"Failed to load analytics dashboard: {e}")
        st.stop()

if __name__ == "__main__":
    main()
