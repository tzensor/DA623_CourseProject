import os
import re
from collections import defaultdict

def rename_audio_files(folder_path):
    # Dictionary to track the counter per label (1-4)
    label_counters = defaultdict(int)
    
    
    for filename in os.listdir(folder_path):
        
        match = re.match(r'^([1-4])\s*\(.*\)(\.\w+)$', filename)
        if match:
            label = match.group(1)
            extension = match.group(2)
            
            # Increment the counter for this label
            label_counters[label] += 1
            count_str = f"{label_counters[label]:03d}"
            
            new_name = f"{label}_Sample_{count_str}{extension}"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        else:
            print(f"Skipped (invalid format): {filename}")

# Example usage
rename_audio_files(r"C:\Users\bhadr\Downloads\DA623_Project\audio_files")
