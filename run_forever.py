import subprocess

while True:
    try:
        subprocess.run(['python', 'body_track_recorder.py'])
    except KeyboardInterrupt:
        print("Program stopped by user.")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        continue