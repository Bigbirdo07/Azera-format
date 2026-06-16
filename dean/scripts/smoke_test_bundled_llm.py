#!/usr/bin/env python3
"""Smoke test script verifying the end-to-end execution path of the bundled Ollama manager.

It starts the server, confirms tags/model presence, sends a tiny prompt to verify
inference, stops the server, and asserts that the process has shut down.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
import json
from pathlib import Path

# Add project root to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlp.local_model_manager import BundledOllamaManager, get_ollama_manager
from nlp.model_prompt import OLLAMA_URL


def main() -> int:
    print("=== Bundled LLM Smoke Test ===")
    
    # 1. Instantiate the manager
    manager = get_ollama_manager()
    binary_path, err = manager._get_binary_path()
    
    if err:
        print(f"Skipping bundled smoke test: {err}")
        print("To run this test, populate binaries inside dean/bin/ and download models.")
        return 0

    print(f"Found binary at: {binary_path}")
    print(f"Target port: {manager.port}")
    print(f"Target URL: {manager.url}")
    
    # Enable fallback override to prevent system conflicts
    os.environ["DEAN_ALLOW_SYSTEM_OLLAMA_FALLBACK"] = "false"
    
    model_name = "llama3.2:3b"
    print(f"Starting server with model: {model_name}...")
    
    # 2. Start the manager
    success = manager.start(model_name=model_name, debug_mode=True)
    status = manager.status
    
    if not success:
        print(f"❌ Failed to start server: {status.error_message}")
        return 1
        
    print(f"✓ Server started. Status mode: {status.mode}")
    print(f"✓ Model available: {status.model_available}")
    
    # 3. Send a tiny prompt to verify inference
    print("Sending test completion request...")
    payload = {
        "model": model_name,
        "prompt": "Reply with a JSON object containing exactly the key 'ok' set to true.",
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_ctx": 1024
        }
    }
    
    try:
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            text = result.get("response", "").strip()
            print(f"Raw model response: {text}")
            parsed = json.loads(text)
            if parsed.get("ok") is True:
                print("✓ Inference verified successfully!")
            else:
                print(f"❌ Unexpected response content: {parsed}")
                return 1
    except Exception as exc:
        print(f"❌ Inference call failed: {exc}")
        manager.stop()
        return 1

    # 4. Stop the manager
    print("Stopping Ollama manager...")
    manager.stop()
    time.sleep(1)
    
    # 5. Assert the process is terminated
    if manager.process is not None and manager.process.poll() is None:
        print("❌ Process did not shut down cleanly.")
        return 1
        
    print("✓ Background process is gone.")
    print("=== Smoke Test Passed! ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
