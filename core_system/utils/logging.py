"""
Simple logging utilities.
"""
import datetime, json, os

def log_event(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.datetime.now().isoformat(),
            "data": data
        }) + "\n")
