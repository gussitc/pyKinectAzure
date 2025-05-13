import subprocess
import time

start_time = time.time()
crash_count = 0

while True:
    subprocess.run(['python', 'body_track_recorder.py'])

    crash_count += 1
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    total_runtime = time.time() - start_time

    print("\n" + "=" * 50)
    print(f"\033[91mAn error occurred at {current_time}.\033[0m")  # Red text
    print(f"\033[96mTotal runtime:\033[0m {total_runtime:.2f} seconds")  # Cyan text
    print(f"\033[92mTotal crashes:\033[0m {crash_count}")  # Green text
    print("=" * 50 + "\n")