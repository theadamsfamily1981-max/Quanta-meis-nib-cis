@echo off
REM TF-A-N Control Center Launcher for Windows

echo ===========================================
echo TF-A-N Control Center
echo ===========================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Warning: Streamlit not installed. Installing...
    pip install streamlit plotly pandas
)

REM Launch GUI
echo Starting Control Center...
echo.
echo The GUI will open in your default browser.
echo If it doesn't, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

streamlit run tfan_gui.py
pause
