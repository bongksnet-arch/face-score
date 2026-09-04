@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo.
echo ============================================
echo   FaceScore Web App Starting...
echo   Browser will open automatically.
echo   To stop: press Ctrl+C in this window
echo ============================================
echo.
"%~dp0venv\Scripts\python.exe" -m streamlit run app.py
pause
