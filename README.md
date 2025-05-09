# DA623_CourseProject
Final Course Project for DA623 (8th Semester IIT Guwahati). Made by Bhadra Tendulkar (4th year B.Tech, Mechanical Engineering, 210103111)

Overview:
This repository contains code for extracting Directed Kernel Pattern (DKP) features from audio signals, particularly for classifying short audio clips like cough sounds or speech events. It performs multi-level pooling, subband decomposition, and pattern extraction using directional kernels, followed by histogram-based feature representation.

Features:
Reads .ogg or .m4a audio files and resamples to 16kHz.
Applies min, max, average, and conditional pooling to generate 40 subbands over 10 levels.
Extracts 41 DKP histograms (1 original + 40 subbands), each of length 1536, resulting in a 62,976-dimensional feature vector.
Outputs a labeled dataset suitable for downstream ML tasks

Dataset: Cough audio for asthma, heart failure, covid and healthy patients, comprising of ogg and m4a files

Working:
Preprocessing:
Audio files are trimmed to even lengths and downsampled as needed.
Subbands are generated using various pooling strategies.

DKP Feature Extraction:
Each subband is broken into overlapping blocks of size 9.
Blocks are reshaped into 3×3 matrices.
Directed kernels apply knight-move comparisons to generate binary patterns.
Binary values are converted to decimal, forming histograms.

Dataset Creation:
Feature vectors are flattened and saved to a CSV file.
Labels are extracted from filenames in the format <label>_<rest>.ogg/m4a
