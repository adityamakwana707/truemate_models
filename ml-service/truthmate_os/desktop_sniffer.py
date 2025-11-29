import time
import pyperclip
from config import settings

def main():
    """
    Feature 15: Runs locally, watches clipboard, prints analysis to console.
    """
    print("--- TruthMate Desktop Watchdog Active ---")
    print("Copy text or links to see analysis...")
    
    last_paste = ""
    
    while True:
        try:
            current = pyperclip.paste()
            if current != last_paste and current.strip():
                last_paste = current
                
                print(f"\n[DETECTED] New Clipboard Content ({len(current)} chars)")
                
                if current.startswith("http"):
                    print(">> Detected Link. Recommendation: Run Link Safety Agent.")
                elif len(current) > 50:
                    print(">> Detected Text Block. Recommendation: Run AI/Propaganda Detector.")
                else:
                    print(">> Content too short for analysis.")
                    
            time.sleep(1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
