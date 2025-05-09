import numpy as np
import librosa
import pandas as pd
from scipy import stats
import os

def read_audio(filepath, sr=16000):
    signal, sample_rate = librosa.load(filepath, sr=sr)
    return signal

def trim_to_even(signal):
    if len(signal) % 2 != 0:
        return signal[:-1]
    return signal

def min_pool(signal):
    signal = trim_to_even(signal)
    return np.min(signal.reshape(-1, 2), axis=1)

def max_pool(signal):
    signal = trim_to_even(signal)
    return np.max(signal.reshape(-1, 2), axis=1)

def avg_pool(signal):
    signal = trim_to_even(signal)
    return np.mean(signal.reshape(-1, 2), axis=1)

def conditional_pool(signal):
    signal = trim_to_even(signal)
    pooled = []
    for i in range(0, len(signal), 2):
        a, b = signal[i], signal[i+1]
        if a > 0 and b > 0:
            pooled.append(max(a, b))
        elif a < 0 and b < 0:
            pooled.append(min(a, b))
        else:
            pooled.append((a + b) / 2)
    return np.array(pooled)

def generate_subbands(signal, levels=10):
    subbands = []
    current = signal.copy()
    for _ in range(levels):
        subbands.extend([
            min_pool(current),
            max_pool(current),
            avg_pool(current),
            conditional_pool(current)
        ])
        current = avg_pool(current)  # Downsample for next level
    return subbands

def generate_dkp_features(signal):
    """
    Generate DKP features from a 1D signal (either original cough sound or a subband)
    
    Args:
        signal: 1D numpy array containing the audio signal or subband
        
    Returns:
        feature_vector: Merged histogram features (1536 features)
    """
    # Step 2.1: Divide the signal into overlapping blocks of size 9
    blocks = []
    for i in range(len(signal) - 8):
        blocks.append(signal[i:i+9])
    
    # Handle edge case if signal is too short
    if len(blocks) == 0:
        # Pad the signal if needed
        padded_signal = np.pad(signal, (0, 9 - len(signal)))
        blocks = [padded_signal]
    
    # Calculate threshold value 'd' (half of the standard deviation)
    d = np.std(signal) / 2
    
    # Initialize arrays to store the map signals for all blocks
    map_signals = [[] for _ in range(6)]
    
    for block in blocks:
        # Step 2.2: Transform the 9-sized vector to a 3x3 matrix
        matrix = block.reshape(3, 3)
        
        # Step 2.3 & 2.4: Apply DKP and generate bits
        # Define the DKP moves (knight patterns)
        # Black knight moves (8 moves)
        black_moves = [
            ((0, 0), (0, 2)),  # 1
            ((0, 2), (2, 2)),  # 2
            ((2, 2), (2, 0)),  # 3
            ((2, 0), (0, 0)),  # 4
            ((0, 1), (1, 2)),  # 5
            ((1, 2), (2, 1)),  # 6
            ((2, 1), (1, 0)),  # 7
            ((1, 0), (0, 1))   # 8
        ]
        
        # Red knight moves (8 moves)
        red_moves = [
            ((1, 1), (0, 0)),  # 1
            ((1, 1), (0, 2)),  # 2
            ((1, 1), (2, 2)),  # 3
            ((1, 1), (2, 0)),  # 4
            ((1, 1), (0, 1)),  # 5
            ((1, 1), (1, 2)),  # 6
            ((1, 1), (2, 1)),  # 7
            ((1, 1), (1, 0))   # 8
        ]
        
        # Combine all moves
        all_moves = black_moves + red_moves
        
        # Initialize bit arrays for each directed kernel
        bits_d1 = []  # directed signum
        bits_d2 = []  # directed upper ternary
        bits_d3 = []  # directed lower ternary
        
        # Apply three directed kernels to generate bits
        for (src_y, src_x), (dst_y, dst_x) in all_moves:
            a = matrix[src_y, src_x]
            b = matrix[dst_y, dst_x]
            
            # Calculate direction based on positions
            dir_x = dst_x - src_x
            dir_y = dst_y - src_y
            
            # Normalize direction to -1, 0, or 1
            if dir_x != 0:
                dir_x = dir_x // abs(dir_x)
            if dir_y != 0:
                dir_y = dir_y // abs(dir_y)
            
            # Directed signum kernel (d1)
            if a > b:
                bits_d1.append(1)
            else:
                bits_d1.append(0)
            
            # Directed upper ternary kernel (d2)
            if a > b + d:
                bits_d2.append(1)
            else:
                bits_d2.append(0)
            
            # Directed lower ternary kernel (d3)
            if a < b - d:
                bits_d3.append(1)
            else:
                bits_d3.append(0)
        
        # Step 2.5: Calculate map signals (convert binary to decimal)
        # For each of the 3 directed kernels, we separate bits for black and red moves
        # This gives us 6 map signals
        
        # Black moves bits (8 bits each)
        black_bits_d1 = bits_d1[:8]
        black_bits_d2 = bits_d2[:8]
        black_bits_d3 = bits_d3[:8]
        
        # Red moves bits (8 bits each)
        red_bits_d1 = bits_d1[8:]
        red_bits_d2 = bits_d2[8:]
        red_bits_d3 = bits_d3[8:]
        
        # Convert binary to decimal
        # We need to reverse the bits because in binary representation,
        # the rightmost bit is the least significant (2^0)
        decimal_values = [
            int("".join(map(str, black_bits_d1[::-1])), 2),
            int("".join(map(str, black_bits_d2[::-1])), 2),
            int("".join(map(str, black_bits_d3[::-1])), 2),
            int("".join(map(str, red_bits_d1[::-1])), 2),
            int("".join(map(str, red_bits_d2[::-1])), 2),
            int("".join(map(str, red_bits_d3[::-1])), 2)
        ]
        
        # Add decimal values to map signals
        for i in range(6):
            map_signals[i].append(decimal_values[i])
    
    # Step 2.6: Extract histograms from map signals
    histograms = []
    for i in range(6):
        # For 8 bits, we have 2^8 = 256 possible values (0-255)
        hist, _ = np.histogram(map_signals[i], bins=256, range=(0, 255))
        histograms.append(hist)
    
    # Step 2.7: Merge histograms
    feature_vector = np.concatenate(histograms)
    
    return feature_vector

def process_audio_with_dkp(original_signal, subbands):
    """
    Process original audio signal and subbands using DKP method
    
    Args:
        original_signal: Original 1D audio signal
        subbands: List of subbands generated from the original signal
        
    Returns:
        all_features: List of 41 feature vectors (1 original + 40 subbands)
    """
    all_features = []
    
    # Process the original signal
    original_features = generate_dkp_features(original_signal)
    all_features.append(original_features)
    
    # Process each subband
    for subband in subbands:
        subband_features = generate_dkp_features(subband)
        all_features.append(subband_features)
    
    return all_features

def extract_label_from_filename(filename):
    base = os.path.basename(filename)
    label_str = base.split("_")[0]
    return int(label_str)

import os
import numpy as np
import pandas as pd

def extract_label_from_filename(filename):
    try:
        return int(filename.split("_")[0])
    except:
        raise ValueError(f"Filename format incorrect for label extraction: {filename}")

def process_audio_folder(folder_path, output_csv="audio_features_with_labels.csv"):
    all_feature_vectors = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".ogg") or filename.endswith(".m4a"):
            full_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")

            try:
                # Load and process audio
                audio_signal = read_audio(full_path, sr=16000)
                audio_signal = trim_to_even(audio_signal)
                audio_subbands = generate_subbands(audio_signal, levels=10)
                feature_vectors = process_audio_with_dkp(audio_signal, audio_subbands)

                # Flatten 41×1536 feature matrix to 1D vector
                combined_vector = np.concatenate(feature_vectors)
                all_feature_vectors.append(combined_vector)

                # Extract label
                label = extract_label_from_filename(filename)
                labels.append(label)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    # Convert to DataFrame
    feature_columns = [f"f{i}" for i in range(len(all_feature_vectors[0]))]
    df = pd.DataFrame(all_feature_vectors, columns=feature_columns)
    df["label"] = labels

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV with shape {df.shape} to: {output_csv}")

    return df

def main():
    folder_path = r"C:\Users\bhadr\Downloads\DA623_Project\audio_files"
    process_audio_folder(folder_path)

if __name__ == "__main__":
    main()