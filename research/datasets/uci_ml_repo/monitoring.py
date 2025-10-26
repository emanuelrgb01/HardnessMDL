import psutil
import time
import datetime
import platform

def show_system_info():
    """Display basic system info."""
    print("=" * 50)
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor() or 'Unknown'}")
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    print(f"CPU cores: {cpu_count} | Threads: {cpu_threads}")
    if cpu_freq:
        print(f"CPU frequency: {cpu_freq.current:.2f} MHz")
    
    # Memory info
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    print(f"Total RAM: {total_gb:.2f} GB")
    print("=" * 50)
    print()

def monitor(interval: int = 5):
    """Monitor CPU and RAM usage every `interval` seconds."""
    show_system_info()
    print(f"{'Time':<20} {'CPU (%)':<10} {'RAM (%)':<10}")
    print("-" * 40)

    try:
        while True:
            cpu_usage = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            ram_usage = ram.percent

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp:<20} {cpu_usage:<10.2f} {ram_usage:<10.2f}")

            time.sleep(interval - 1)  # adjust for total loop time

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    monitor(interval=5)
