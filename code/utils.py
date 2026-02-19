import time
def print_time(*args, **kwargs):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]: ", end="")
    print(*args, **kwargs)