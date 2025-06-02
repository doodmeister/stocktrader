---
applyTo: '**'
---

# StockTrader Development Guidelines

## Environment & Platform Standards

### Operating System & Shell Requirements
- **Target OS**: Windows
- **Default Shell**: Bash (`bash.exe`)
- **All terminal commands MUST be bash-compatible**

### Terminal Command Standards
Do not use emojis in any terminal commands or code snippets. Use clear, concise bash commands that are compatible with the Windows environment.
#### ✅ Use Bash Syntax
```bash
# Correct: Bash commands
pwd
cd /c/dev/stocktrader
source venv/Scripts/activate
streamlit run main.py
python -m pip install -r requirements.txt
```

#### ❌ Avoid PowerShell Commands
```powershell
# Incorrect: Do not use PowerShell syntax
Get-Location
Set-Location c:\dev\stocktrader
.\venv\Scripts\activate
```

### Project Status & Architecture

#### ✅ Modular Architecture COMPLETED (May 29, 2025)
- **Status**: 100% Complete and Functional
- **Entry Point**: `streamlit run main.py` (modular)
- **Legacy**: `streamlit_dashboard.py` (deprecated, shows migration notice)

#### Core Modules (All Completed ✅)
1. **`main.py`** - Modular dashboard entry point
2. **`core/dashboard_controller.py`** - UI orchestration and navigation  
3. **`core/page_loader.py`** - Dynamic page discovery and management
4. **`core/health_checks.py`** - System health monitoring with 30s caching
5. **`core/session_manager.py`** - Manages user sessions and state

#### Project Commands (Bash)
```bash
# Environment setup
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# Launch dashboard (recommended)
streamlit run main.py

# Development commands
python -c "import main; print('Imports working')"
python -c "from core import dashboard_controller; print('Core modules working')"

# Testing
python -m pytest tests/
find core/ -name "*.py" -exec basename {} .py \; | while read module; do python -c "import core.$module; print('$module.py imported successfully')"; done
```

## Development Standards

### File and Module Guidelines
- **Module Organization**: Use the established `core/`, `dashboard_pages/`, `utils/` structure
- **Import Paths**: Use absolute imports (`from core.health_checks import HealthChecker`)
- **Error Handling**: Leverage the modular error isolation system
- **Caching**: Utilize the 30-second TTL caching system for performance

### Code Quality Standards
- **Function Length**: Keep modules focused (~300 lines max)
- **Single Responsibility**: Each module should have one clear purpose
- **Type Hints**: Use Python type annotations where applicable
- **Docstrings**: Google-style docstrings for all public functions
- **Error Messages**: Provide clear, actionable error messages

### Documentation Standards
- **Status Indicators**: Use ✅ for completed features, 🚀 for new features, ⚠️ for deprecated
- **Command Examples**: Always use bash syntax
- **Architecture Notes**: Reference the modular structure in explanations
- **Completion Status**: Always acknowledge that modularization is complete

## Security & Best Practices

### Path Handling
```bash
# Correct: Windows paths in bash format
projectPath="/c/dev/stocktrader"
corePath="$projectPath/core"
```

### Environment Variables
```bash
# Bash environment variable syntax
export PYTHONPATH="/c/dev/stocktrader"
export STREAMLIT_SERVER_PORT="8501"
```

### File Operations
```bash
# Use bash commands
cp "source/file.py" "destination/file.py"
rm -f temp/*.log
mkdir -p new_folder
```

## AI Assistant Guidelines

### When Providing Instructions
1. **Always use bash commands** for Windows environment
2. **Reference completed modular architecture** - don't suggest recreating existing modules
3. **Use absolute file paths** with Unix-style forward slashes
4. **Acknowledge completion status** of the modularization project
5. **Provide working examples** that can be copy-pasted into bash

### Response Format
- **Commands**: Use bash syntax with proper escaping
- **File Paths**: Use Unix format (`/c/dev/stocktrader/core/file.py`)
- **Status**: Always acknowledge completed modular architecture
- **Examples**: Provide runnable bash commands

### Completion Acknowledgment
Always recognize that:
- ✅ Modular architecture is 100% complete and functional
- ✅ All 4 core modules are implemented and tested
- ✅ Health monitoring system is active with caching
- ✅ Dashboard runs successfully on Windows with bash
- ✅ Import system is fixed and working properly