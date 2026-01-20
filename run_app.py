import sys
import threading
import time
import webbrowser
from streamlit.web import cli as stcli
from streamlit.runtime import get_instance

HOST = "127.0.0.1"
PORT = 8502
URL = f"http://{HOST}:{PORT}"


def open_browser():
    time.sleep(1.5)
    webbrowser.open(URL)


def shutdown_when_no_sessions(poll_seconds: float = 2.0, grace_seconds: float = 5.0):
    """
    Periodically checks Streamlit runtime.
    If there are no active sessions for `grace_seconds`, exit the process.
    """
    rt = None
    start_empty = None

    while rt is None:
        try:
            rt = get_instance()
        except Exception:
            time.sleep(0.5)

    while True:
        sessions = rt._session_mgr.list_sessions()
        if len(sessions) == 0:
            if start_empty is None:
                start_empty = time.time()
            elif time.time() - start_empty >= grace_seconds:
                sys.exit(0)
        else:
            start_empty = None

        time.sleep(poll_seconds)


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    threading.Thread(target=shutdown_when_no_sessions, daemon=True).start()

    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        "--server.headless=true",
        f"--server.address={HOST}",
        f"--server.port={PORT}",
        "--browser.gatherUsageStats=false",
    ]

    sys.exit(stcli.main())
