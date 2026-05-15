@echo off
echo ========================================
echo  Starting Anomaly Detection System
echo ========================================
echo.
echo Opening in browser...
echo.
cd /d "%~dp0"
set PYTHONPATH=%~dp0
"C:\Users\SZHH59\AppData\Local\anaconda3\envs\py310\python.exe" -m streamlit run app.py
pause
