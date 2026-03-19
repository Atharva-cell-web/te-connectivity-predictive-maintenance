"""
Start Server Script
Kills any zombie process holding the target port, then starts Uvicorn.
Usage:  python start_server.py [--port 8080]
"""
import argparse
import os
import subprocess
import sys
import time


def kill_port(port: int) -> bool:
    """Attempt to free the given port on Windows by killing the holding process."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
                pid = int(parts[4])
                print(f"[start_server] Port {port} held by PID {pid} — killing…")
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], timeout=10)
                time.sleep(1)
                return True
    except Exception as exc:
        print(f"[start_server] Warning while checking port: {exc}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Start the Uvicorn backend server.")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    kill_port(args.port)

    print(f"[start_server] Starting Uvicorn on {args.host}:{args.port} …")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    sys.exit(
        subprocess.call([
            sys.executable, "-m", "uvicorn",
            "api:app",
            "--host", args.host,
            "--port", str(args.port),
            "--reload",
        ], env=env)
    )


if __name__ == "__main__":
    main()
