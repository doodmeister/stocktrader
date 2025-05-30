---
applyTo: '**'
---

# StockTrader Development Guidelines

## Environment & Platform Standards

### Operating System & Shell Requirements
- **Target OS**: Windows
- **Default Shell**: PowerShell (`powershell.exe`)
- **All terminal commands MUST be PowerShell-compatible**

### Terminal Command Standards

#### ✅ Use PowerShell Syntax
```powershell
# Correct: PowerShell commands
Get-Location
Set-Location c:\dev\stocktrader
.\venv\Scripts\activate
streamlit run main.py
python -m pip install requirements.txt
```

#### ❌ Avoid Bash/Unix Commands
```bash
# Incorrect: Do not use bash syntax
pwd
cd /dev/stocktrader
source venv/bin/activate
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

#### Project Commands (PowerShell)
```powershell
# Environment setup
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Launch dashboard (recommended)
streamlit run main.py

# Development commands
python -c "import main; print('Imports working')"
python -c "from core import dashboard_controller; print('Core modules working')"

# Testing
python -m pytest tests\
Get-ChildItem -Path "core\" -Filter "*.py" | ForEach-Object { python -c "import core.$($_.BaseName); print('$($_.Name) imported successfully')" }
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
- **Command Examples**: Always use PowerShell syntax
- **Architecture Notes**: Reference the modular structure in explanations
- **Completion Status**: Always acknowledge that modularization is complete

## Security & Best Practices

### Path Handling
```powershell
# Correct: Windows path separators
$projectPath = "c:\dev\stocktrader"
$corePath = "$projectPath\core"
```

### Environment Variables
```powershell
# PowerShell environment variable syntax
$env:PYTHONPATH = "c:\dev\stocktrader"
$env:STREAMLIT_SERVER_PORT = "8501"
```

### File Operations
```powershell
# Use PowerShell cmdlets
Copy-Item "source\file.py" -Destination "destination\file.py"
Remove-Item "temp\*.log" -Force
New-Item -ItemType Directory -Path "new_folder" -Force
```

## AI Assistant Guidelines

### When Providing Instructions
1. **Always use PowerShell commands** for Windows environment
2. **Reference completed modular architecture** - don't suggest recreating existing modules
3. **Use absolute file paths** with Windows-style backslashes
4. **Acknowledge completion status** of the modularization project
5. **Provide working examples** that can be copy-pasted into PowerShell

### Response Format
- **Commands**: Use PowerShell syntax with proper escaping
- **File Paths**: Use Windows format (`c:\dev\stocktrader\core\file.py`)
- **Status**: Always acknowledge completed modular architecture
- **Examples**: Provide runnable PowerShell commands

### Completion Acknowledgment
Always recognize that:
- ✅ Modular architecture is 100% complete and functional
- ✅ All 4 core modules are implemented and tested
- ✅ Health monitoring system is active with caching
- ✅ Dashboard runs successfully on Windows with PowerShell
- ✅ Import system is fixed and working properly