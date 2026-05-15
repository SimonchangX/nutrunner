"""
Launcher for standalone Atlas Copco Anomaly Detection Dashboard
Built with PyInstaller for distribution without Python environment
"""

import sys
import os
import subprocess
import tempfile
import shutil
import socket

def find_free_port():
    """Find an available port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def main():
    # Determine paths
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        base_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        base_dir = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(base_dir, 'app.py')

    if not os.path.exists(app_path):
        # Try to extract from bundled resources
        import streamlit
        streamlit_dir = os.path.join(base_dir, '_internal')
        app_path = os.path.join(streamlit_dir, 'app.py')

    if not os.path.exists(app_path):
        print(f"ERROR: app.py not found at {app_path}")
        input("Press Enter to exit...")
        return

    # Find free port and run Streamlit
    port = find_free_port()
    print(f"Starting Atlas Copco Dashboard on http://localhost:{port}")
    print("The dashboard will open in your browser...")

    # Try to open browser
    import webbrowser
    webbrowser.open(f'http://localhost:{port}')

    # Run Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', app_path,
        '--server.port', str(port),
        '--server.headless', 'true',
        '--browser.serverAddress', 'localhost',
        '--browser.gatherUsageStats', 'false',
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        input("Press Enter to exit...")

if __name__ == '__main__':
    main()
