#!/bin/bash
# TF-A-N Control Center Launcher

echo "==========================================="
echo "TF-A-N Control Center"
echo "==========================================="
echo ""

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "⚠ Streamlit not installed. Installing..."
    pip install streamlit plotly pandas
fi

# Launch GUI
echo "🚀 Launching Control Center..."
echo ""
echo "The GUI will open in your default browser."
echo "If it doesn't, go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

streamlit run tfan_gui.py
