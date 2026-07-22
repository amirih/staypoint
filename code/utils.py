import builtins
import time
import json
import os
from datetime import datetime

# def print_time(*args, **kwargs):
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]: ", end="")
#     print(*args, **kwargs)

def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    


def print_time(*args, **kwargs):
    builtins._original_print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]: ",
        *args,
        **kwargs
    )

def install():
    if not hasattr(builtins, "_original_print"):
        builtins._original_print = builtins.print
    builtins.print = print_time