# Streamlit Dashboard Template - Quick Reference

## 🚀 Quick Start Checklist

### 1. Copy Template

```bash
cp templates/streamlit_dashboard_template.py dashboard_pages/my_dashboard.py
```

### 2. Rename Core Elements

- [ ] Change class name: `TemplateDashboard` → `MyDashboard`
- [ ] Update `PAGE_CONFIG` with your title and icon
- [ ] Modify `SESSION_STATE_KEYS` for your data needs
- [ ] Update imports for your specific requirements

### 3. Implement Core Methods

- [ ] `render_main_content()` - Your main dashboard logic
- [ ] Custom configuration in `TemplateConfig`
- [ ] Custom validation in `TemplateValidator`
- [ ] Error handling for domain-specific issues

## 📋 Essential Code Patterns

### Basic Dashboard Structure

```python
class MyDashboard(TemplateDashboard):
    def __init__(self):
        super().__init__()
        # Your initialization
    
    @handle_dashboard_errors
    def render_main_content(self):
        st.header("My Dashboard")
        # Your implementation
```

### Configuration Management

```python
class MyConfig(TemplateConfig):
    def __init__(self):
        super().__init__()
        self.config.update({
            "my_setting": "value",
            "api_url": "https://api.example.com"
        })
```

### Custom Validation

```python
class MyValidator(TemplateValidator):
    @staticmethod
    def validate_my_data(data):
        if not data:
            raise DashboardException("Data required")
        return True
```

### Error Handling

```python
@handle_dashboard_errors
def my_method(self):
    try:
        # Your code
        result = risky_operation()
        return result
    except MyCustomError as e:
        logger.error(f"Custom error: {e}")
        st.error(f"Operation failed: {e}")
```

## 🔧 Common Customizations

### Session State Setup

```python
CUSTOM_SESSION_KEYS = {
    **SESSION_STATE_KEYS,  # Include base keys
    "my_data": None,
    "my_settings": {},
    "my_cache": {}
}
```

### Sidebar Controls

```python
def render_sidebar(self):
    with st.sidebar:
        st.header("My Controls")
        
        value = st.slider("Setting", 0, 100, 50)
        option = st.selectbox("Option", ["A", "B", "C"])
        
        # Update session state
        st.session_state.my_settings = {
            "value": value,
            "option": option
        }
    
    # Include parent sidebar
    super().render_sidebar()
```

### Data Processing

```python
def process_data(self, df):
    try:
        # Validate input
        self.validator.validate_dataframe(df)
        
        # Process data
        processed_df = df.copy()
        # Your processing logic
        
        # Store in session state
        st.session_state.my_data = processed_df
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise DashboardException(f"Processing error: {e}")
```

## 📊 Visualization Patterns

### Basic Chart

```python
@handle_dashboard_errors
def render_chart(self, df, column):
    fig = px.line(df, x='date', y=column, title=f"{column} Over Time")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
```

### Metrics Display

```python
def render_metrics(self, data):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total", f"{data['total']:,}")
    with col2:
        st.metric("Average", f"{data['avg']:.2f}")
    with col3:
        st.metric("Growth", f"{data['growth']:.1%}")
```

### Interactive Controls

```python
def render_controls(self):
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.date_input("Date Range", value=[start, end])
    
    with col2:
        metrics = st.multiselect("Metrics", options=available_metrics)
    
    return date_range, metrics
```

## 🔒 Security Best Practices

### File Upload Validation

```python
def handle_file_upload(self):
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            # Validate file
            self.validator.validate_file_upload(uploaded_file)
            
            # Process file
            df = pd.read_csv(uploaded_file)
            self.validator.validate_dataframe(df)
            
            return df
            
        except DashboardException as e:
            st.error(f"File error: {e}")
            return None
```

### Input Sanitization

```python
def validate_user_input(self, value, min_val=0, max_val=1000):
    try:
        self.validator.validate_numeric_input(value, min_val, max_val)
        return value
    except DashboardException as e:
        st.error(f"Invalid input: {e}")
        return None
```

## ⚡ Performance Optimization

### Caching Expensive Operations

```python
@st.cache_data(ttl=300)  # Cache for 5 minutes
def expensive_computation(data):
    # Your expensive operation
    return processed_data

@st.cache_resource
def load_model():
    # Load ML model or other resources
    return model
```

### Data Streaming for Large Datasets

```python
def process_large_dataset(self, df):
    chunk_size = self.config.get("chunk_size", 1000)
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        yield self.process_chunk(chunk)
```

## 🐛 Debugging Tips

### Enable Debug Mode

```python
# In configuration
"show_debug_info": True

# In sidebar
if self.config.get("show_debug_info"):
    with st.expander("Debug Info"):
        st.write("Session State:", dict(st.session_state))
        st.write("Config:", self.config.config)
```

### Performance Monitoring

```python
# Automatically tracked by @handle_dashboard_errors decorator
# View in footer when show_performance_metrics=True

# Manual timing
start_time = time.time()
# Your operation
execution_time = time.time() - start_time
st.write(f"Operation took {execution_time:.3f}s")
```

### Error Logging

```python
try:
    # Your operation
    pass
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    st.error("Operation failed - check logs")
```

## 🧪 Testing Patterns

### Unit Testing

```python
def test_dashboard_initialization():
    dashboard = MyDashboard()
    assert dashboard.config is not None
    assert dashboard.validator is not None

def test_data_validation():
    dashboard = MyDashboard()
    test_df = pd.DataFrame({"col1": [1, 2, 3]})
    assert dashboard.validator.validate_dataframe(test_df)
```

### Integration Testing

```python
@patch('streamlit.session_state', new_callable=dict)
def test_session_state_setup(mock_session_state):
    dashboard = MyDashboard()
    dashboard._initialize_session_state()
    assert "my_data" in mock_session_state
```

## 📁 File Organization

``` python
dashboard_pages/
├── my_dashboard.py              # Your main dashboard
├── components/
│   ├── my_charts.py            # Custom chart components
│   ├── my_controls.py          # Custom control components
│   └── my_validators.py        # Custom validation logic
├── config/
│   ├── my_dashboard_config.py  # Dashboard-specific config
│   └── constants.py            # Constants and enums
└── tests/
    ├── test_my_dashboard.py    # Dashboard tests
    └── test_components.py      # Component tests
```

## 🚀 Deployment Checklist

### Production Readiness

- [ ] Error handling implemented for all user interactions
- [ ] Input validation for all user inputs
- [ ] Performance monitoring enabled
- [ ] Logging configured properly
- [ ] Security validations in place
- [ ] Configuration externalized
- [ ] Health checks implemented
- [ ] Documentation updated

### Environment Configuration

```python
# Environment-specific settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    CONFIG.update({
        "show_debug_info": False,
        "enable_profiling": False,
        "log_level": "WARNING"
    })
```

## 📞 Support Resources

1. **Template Documentation**: `templates/DASHBOARD_TEMPLATE_DOCS.md`
2. **Example Implementation**: `templates/example_analytics_dashboard.py`
3. **Core Utilities**: `core/dashboard_utils.py`
4. **Logger Configuration**: `utils/logger.py`
5. **Existing Dashboards**: `dashboard_pages/` for reference patterns

## 🔄 Template Updates

When updating the template:

1. Update version number in template comments
2. Add new features to this quick reference
3. Update example implementations
4. Test with existing dashboards
5. Document breaking changes
6. Notify team of updates
