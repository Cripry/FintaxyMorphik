import argparse
import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time

import requests
import tomli
import uvicorn

from core.config import get_settings
from core.logging_config import setup_logging
from utils.env_loader import load_local_env

# Global variables to store subprocess handles
worker_process = None
ui_process = None


def wait_for_redis(host="localhost", port=6379, timeout=20):
    """
    Wait for Redis to become available.

    Args:
        host: Redis host address
        port: Redis port number
        timeout: Maximum time to wait in seconds

    Returns:
        True if Redis becomes available within the timeout, False otherwise
    """
    logging.info(f"Waiting for Redis to be available at {host}:{port}...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                logging.info("Redis is accepting connections.")
                return True
        except (OSError, socket.error):
            logging.debug(f"Redis not available yet, retrying... ({int(time.monotonic() - t0)}s elapsed)")
            time.sleep(0.3)

    logging.error(f"Redis not reachable after {timeout}s")
    return False


def check_and_start_redis():
    """Check if the Redis container is running, start if necessary."""
    try:
        # Check if container exists and is running
        check_running_cmd = ["docker", "ps", "-q", "-f", "name=morphik-redis"]
        running_container = subprocess.check_output(check_running_cmd).strip()

        if running_container:
            logging.info("Redis container (morphik-redis) is already running.")
            return

        # Check if container exists but is stopped
        check_exists_cmd = ["docker", "ps", "-a", "-q", "-f", "name=morphik-redis"]
        existing_container = subprocess.check_output(check_exists_cmd).strip()

        if existing_container:
            logging.info("Starting existing Redis container (morphik-redis)...")
            subprocess.run(["docker", "start", "morphik-redis"], check=True, capture_output=True)
            logging.info("Redis container started.")
        else:
            logging.info("Creating and starting Redis container (morphik-redis)...")
            subprocess.run(
                ["docker", "run", "-d", "--name", "morphik-redis", "-p", "6379:6379", "redis"],
                check=True,
                capture_output=True,
            )
            logging.info("Redis container created and started.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to manage Redis container: {e}")
        logging.error(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("Docker command not found. Please ensure Docker is installed and in PATH.")
        sys.exit(1)


def start_arq_worker():
    """Start the ARQ worker as a subprocess."""
    global worker_process
    try:
        logging.info("Starting ARQ worker...")

        # Ensure logs directory exists
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Worker log file paths
        worker_log_path = os.path.join(log_dir, "worker.log")

        # Open log files
        worker_log = open(worker_log_path, "a")

        # Add timestamp to log (use Python for cross-platform compatibility)
        from datetime import datetime

        timestamp = datetime.now().isoformat(timespec="seconds")
        worker_log.write(f"\n\n--- Worker started at {timestamp} ---\n\n")
        worker_log.flush()

        # Use sys.executable to ensure the same Python environment is used
        worker_cmd = [sys.executable, "-m", "arq", "core.workers.ingestion_worker.WorkerSettings"]

        # Start the worker with output redirected to log files
        worker_process = subprocess.Popen(
            worker_cmd,
            stdout=worker_log,
            stderr=worker_log,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),  # Ensure unbuffered output
        )
        logging.info(f"ARQ worker started with PID: {worker_process.pid}")
        logging.info(f"Worker logs available at: {worker_log_path}")
    except Exception as e:
        logging.error(f"Failed to start ARQ worker: {e}")
        sys.exit(1)


def start_ui():
    """Start the Morphik UI frontend as a subprocess."""
    global ui_process
    try:
        ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ee", "ui-component")
        if not os.path.isdir(ui_dir):
            logging.warning(f"UI directory not found at {ui_dir}, skipping UI startup.")
            return

        logging.info("Starting Morphik UI...")

        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        ui_log_path = os.path.join(log_dir, "ui.log")
        ui_log = open(ui_log_path, "a")

        from datetime import datetime
        timestamp = datetime.now().isoformat(timespec="seconds")
        ui_log.write(f"\n\n--- UI started at {timestamp} ---\n\n")
        ui_log.flush()

        # Read the API port from morphik.toml to pass to the UI
        with open("morphik.toml", "rb") as f:
            config = tomli.load(f)
        api_port = config.get("api", {}).get("port", 8000)

        ui_env = dict(
            os.environ,
            NEXT_PUBLIC_API_BASE_URL=f"http://localhost:{api_port}",
        )

        # Use npm.cmd on Windows
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        ui_process = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=ui_dir,
            stdout=ui_log,
            stderr=ui_log,
            env=ui_env,
        )
        logging.info(f"Morphik UI started with PID: {ui_process.pid}")
        logging.info(f"UI available at: http://localhost:3000")
        logging.info(f"UI logs available at: {ui_log_path}")
    except Exception as e:
        logging.warning(f"Failed to start Morphik UI: {e}")


def cleanup_processes():
    """Stop the ARQ worker and UI processes on exit."""
    global worker_process, ui_process
    if worker_process and worker_process.poll() is None:  # Check if process is still running
        logging.info(f"Stopping ARQ worker (PID: {worker_process.pid})...")

        # Log the worker termination
        try:
            log_dir = os.path.join(os.getcwd(), "logs")
            worker_log_path = os.path.join(log_dir, "worker.log")

            from datetime import datetime

            with open(worker_log_path, "a") as worker_log:
                timestamp = datetime.now().isoformat(timespec="seconds")
                worker_log.write(f"\n\n--- Worker stopping at {timestamp} ---\n\n")
        except Exception as e:
            logging.warning(f"Could not write worker stop message to log: {e}")

        # Send SIGTERM first for graceful shutdown
        worker_process.terminate()
        try:
            # Wait a bit for graceful shutdown
            worker_process.wait(timeout=5)
            logging.info("ARQ worker stopped gracefully.")
        except subprocess.TimeoutExpired:
            logging.warning("ARQ worker did not terminate gracefully, sending SIGKILL.")
            worker_process.kill()  # Force kill if it doesn't stop
            logging.info("ARQ worker killed.")

        # Close any open file descriptors for the process
        if hasattr(worker_process, "stdout") and worker_process.stdout:
            worker_process.stdout.close()
        if hasattr(worker_process, "stderr") and worker_process.stderr:
            worker_process.stderr.close()

    # Stop UI process
    if ui_process and ui_process.poll() is None:
        logging.info(f"Stopping Morphik UI (PID: {ui_process.pid})...")
        ui_process.terminate()
        try:
            ui_process.wait(timeout=5)
            logging.info("Morphik UI stopped gracefully.")
        except subprocess.TimeoutExpired:
            logging.warning("Morphik UI did not terminate gracefully, sending SIGKILL.")
            ui_process.kill()
            logging.info("Morphik UI killed.")

    # Optional: Add Redis container stop logic here if desired
    # try:
    #     logging.info("Stopping Redis container...")
    #     subprocess.run(["docker", "stop", "morphik-redis"], check=False, capture_output=True)
    # except Exception as e:
    #     logging.warning(f"Could not stop Redis container: {e}")


# Register the cleanup function to be called on script exit
atexit.register(cleanup_processes)
# Also register for SIGINT (Ctrl+C) and SIGTERM
signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))


def check_ollama_running(base_url):
    """Check if Ollama is running and accessible at the given URL."""
    try:
        api_url = f"{base_url}/api/tags"
        response = requests.get(api_url, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_ollama_usage_info():
    """Check if Ollama is required based on the configuration file and get base URLs."""
    try:
        with open("morphik.toml", "rb") as f:
            config = tomli.load(f)

        ollama_configs = []

        # Get registered Ollama models first
        ollama_models = {}
        if "registered_models" in config:
            for model_key, model_config in config["registered_models"].items():
                model_name = model_config.get("model_name", "")
                if "ollama" in model_name:
                    api_base = model_config.get("api_base")
                    if api_base:
                        ollama_models[model_key] = api_base

        # Check which components are using Ollama models
        components_to_check = ["embedding", "completion", "parser.vision"]

        for component in components_to_check:
            if component == "parser.vision":
                # Special handling for parser.vision
                if "parser" in config and "vision" in config["parser"]:
                    model_key = config["parser"]["vision"].get("model")
                    if model_key in ollama_models:
                        ollama_configs.append({"component": component, "base_url": ollama_models[model_key]})
            else:
                # Standard component check
                if component in config:
                    model_key = config[component].get("model")
                    if model_key in ollama_models:
                        ollama_configs.append({"component": component, "base_url": ollama_models[model_key]})

        # Add contextual chunking model check
        if (
            "parser" in config
            and config["parser"].get("use_contextual_chunking")
            and "contextual_chunking_model" in config["parser"]
        ):
            model_key = config["parser"]["contextual_chunking_model"]
            if model_key in ollama_models:
                ollama_configs.append(
                    {
                        "component": "parser.contextual_chunking",
                        "base_url": ollama_models[model_key],
                    }
                )

        return ollama_configs
    except Exception as e:
        logging.error(f"Error checking Ollama configuration: {e}")
        return []


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start the Morphik server")
    parser.add_argument(
        "--log",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Set the logging level",
    )
    parser.add_argument(
        "--skip-ollama-check",
        action="store_true",
        help="Skip Ollama availability check",
    )
    parser.add_argument(
        "--skip-redis-check",
        action="store_true",
        help="Skip Redis container management (useful when running in Docker)",
    )
    parser.add_argument(
        "--skip-ui",
        action="store_true",
        help="Skip starting the Morphik UI frontend",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for Uvicorn (default: CPU count)",
    )
    args = parser.parse_args()

    # Set up logging first with specified level
    setup_logging(log_level=args.log.upper())

    # Check and start Redis container (unless skipped)
    if not args.skip_redis_check:
        check_and_start_redis()

    # Load environment variables from .env file if secrets aren't injected
    load_local_env(override=True)

    # Check if Ollama is required and running
    if not args.skip_ollama_check:
        ollama_configs = get_ollama_usage_info()

        if ollama_configs:
            # Group configs by base_url to avoid redundant checks
            base_urls = {}
            for config in ollama_configs:
                if config["base_url"] not in base_urls:
                    base_urls[config["base_url"]] = []
                base_urls[config["base_url"]].append(config["component"])

            all_running = True
            for base_url, components in base_urls.items():
                if not check_ollama_running(base_url):
                    print(f"ERROR: Ollama is not accessible at {base_url}")
                    print(f"This URL is used by these components: {', '.join(components)}")
                    all_running = False

            if not all_running:
                print("\nPlease ensure Ollama is running at the configured URLs before starting the server")
                print("Run with --skip-ollama-check to bypass this check")
                sys.exit(1)
            else:
                component_list = [config["component"] for config in ollama_configs]
                print(f"Ollama is running and will be used for: {', '.join(component_list)}")

    # Pre-load Ollama models into VRAM (warmup)
    if not args.skip_ollama_check:
        ollama_configs = get_ollama_usage_info()
        warmup_models = set()
        try:
            with open("morphik.toml", "rb") as f:
                toml_config = tomli.load(f)
            registered = toml_config.get("registered_models", {})
            for cfg in ollama_configs:
                component = cfg["component"]
                base_url = cfg["base_url"]
                # Find the model key for this component
                if component == "parser.vision":
                    model_key = toml_config.get("parser", {}).get("vision", {}).get("model")
                elif component == "parser.contextual_chunking":
                    model_key = toml_config.get("parser", {}).get("contextual_chunking_model")
                else:
                    model_key = toml_config.get(component, {}).get("model")
                if model_key and model_key in registered:
                    model_name = registered[model_key].get("model_name", "")
                    # Strip ollama_chat/ or ollama/ prefix for the API call
                    for prefix in ("ollama_chat/", "ollama/"):
                        if model_name.startswith(prefix):
                            model_name = model_name[len(prefix):]
                            break
                    warmup_models.add((base_url, model_name))
            for base_url, model_name in warmup_models:
                logging.info(f"Warming up Ollama model: {model_name} at {base_url}")
                try:
                    resp = requests.post(
                        f"{base_url}/api/generate",
                        json={"model": model_name, "prompt": "hi", "options": {"num_predict": 1, "num_ctx": 8192}, "stream": False},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        logging.info(f"Model {model_name} loaded into VRAM successfully")
                    else:
                        logging.warning(f"Warmup for {model_name} returned status {resp.status_code}")
                except Exception as e:
                    logging.warning(f"Warmup for {model_name} failed: {e}")
        except Exception as e:
            logging.warning(f"Model warmup skipped: {e}")

    # Load settings (this will validate all required env vars)
    settings = get_settings()

    # Wait for Redis to be available
    if not wait_for_redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT):
        logging.error("Cannot start server without Redis. Please ensure Redis is running.")
        sys.exit(1)

    # Start ARQ worker in the background (delayed to avoid CUDA segfault on Windows)
    import threading
    def _delayed_arq_start():
        import time
        time.sleep(30)  # Wait for main process to finish CUDA/ColPali init
        start_arq_worker()
    threading.Thread(target=_delayed_arq_start, daemon=True).start()

    # Start Morphik UI in the background
    if not args.skip_ui:
        start_ui()

    # Start server (this is blocking)
    logging.info("Starting Uvicorn server...")
    uvicorn.run(
        "core.api:app",
        host=settings.HOST,
        port=settings.PORT,
        loop="asyncio",
        log_level=args.log,
        workers=args.workers,
        # reload=settings.RELOAD # Reload might interfere with subprocess management
    )


if __name__ == "__main__":
    main()
