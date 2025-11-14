import subprocess
import time

start_time = time.time()
crash_count = 0

while True:
    subprocess.run(['python', 'body_track_recorder.py', '--flip'])

    crash_count += 1
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    total_runtime = time.time() - start_time

    print("\n" + "=" * 50)
    print(f"\033[91mAn error occurred at {current_time}.\033[0m")  # Red text
    print(f"\033[96mTotal runtime:\033[0m {total_runtime / 60:.2f} minutes")  # Cyan text
    print(f"\033[92mTotal crashes:\033[0m {crash_count}")  # Green text
    if crash_count > 0:
        avg_time_per_crash = total_runtime / crash_count / 60
        print(f"\033[93mAverage time per crash:\033[0m {avg_time_per_crash:.2f} minutes")  # Yellow text
    print("=" * 50 + "\n")