# OpenAI API Key Setup Guide

## Issue Fixed
The "ChatGPT Insight" feature was failing because the OpenAI API key was not properly configured for the new OpenAI library (v1.x).

## What Was Fixed
1. **Updated `utils/chatgpt.py`**:
   - Changed from legacy `import openai` to `from openai import OpenAI`
   - Added proper client initialization with API key
   - Added better error handling for missing API key

2. **Updated `dashboard_pages/data_analysis_v2.py`**:
   - Removed deprecated `openai.api_key = get_openai_api_key()` line
   - Removed unused `import openai`

## How to Set Up OpenAI API Key

### Option 1: Environment Variable (Recommended)
Set the `OPENAI_API_KEY` environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Windows (Permanent):**
1. Search for "Environment Variables" in Windows
2. Click "Edit the system environment variables"
3. Click "Environment Variables" button
4. Under "User variables", click "New"
5. Variable name: `OPENAI_API_KEY`
6. Variable value: your API key
7. Click OK

### Option 2: .env File
Create a `.env` file in the project root directory:
```
OPENAI_API_KEY=your-api-key-here
```

### How to Get an OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key and use it in one of the methods above

## Testing the Fix
1. Set up your API key using one of the methods above
2. Run the dashboard: `streamlit run dashboard_pages\data_analysis_v2.py`
3. Upload a CSV file
4. Scroll down and click "Get ChatGPT Insight"
5. You should now receive an AI-generated analysis instead of an error

## Error Handling
If you still get an error, the updated code will show:
- "Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable."

This makes it clear what needs to be configured.
