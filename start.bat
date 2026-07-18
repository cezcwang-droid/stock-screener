@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python -m streamlit run stock_screener.py --server.port 8501 --browser.gatherUsageStats false
pause
