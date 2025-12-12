import sys
from pathlib import Path
import pandas as pd
import os
import shutil

#!/usr/bin/env python3
"""
Data filtering script for accented English speech recognition.
"""

def filter_train_data():

    # Load your metadata
    train_metadata_df = pd.read_csv('mcv-scripted-en-v23.0/cv-corpus-23.0-2025-09-05/en/train.tsv', sep='\t')

    # Filter by accents
    accents = ['india', 'zimbabwe', 'united states', 'hong kong']
    def matches_target(accent_value):
        if pd.isna(accent_value):
            return False
        
        accent_lower = str(accent_value).lower()
        return any(keyword in accent_lower for keyword in accents)
    
    train_metadata_specific_accents_df = train_metadata_df[train_metadata_df['accents'].apply(matches_target)]


    # Create the destination directory if it doesn't exist
    dest_dir = 'filtered_train_data'
    os.makedirs(dest_dir, exist_ok=True)

    # Get the base directory where the original files are located
    base_dir = 'mcv-scripted-en-v23.0/cv-corpus-23.0-2025-09-05/en/clips'

    # Get the corresponding audio files of the filtered dataset and copy them to the destination directory
    for i, file_path in enumerate(train_metadata_specific_accents_df['path']):
        src_path = os.path.join(base_dir, file_path)
        dest_path = os.path.join(dest_dir, os.path.basename(file_path))
        print(i)
        try:
            shutil.copy2(src_path, dest_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error copying {src_path}: {str(e)}")

    
    # Ensure all audio files from the filtered metadata exist
    from pathlib import Path

    AUDIO_DIR = "filtered_test_data"

    def check_file_exists(row):
        audio_path = Path(AUDIO_DIR) / row['path']
        return audio_path.exists()

    train_metadata_specific_accents_df = train_metadata_specific_accents_df[train_metadata_specific_accents_df.apply(check_file_exists, axis=1)]
    
    # Save the filtered metadata to a new TSV file
    train_metadata_specific_accents_df.to_csv('./train_data_filtered_accents.tsv', sep='\t', index=False)

def filter_test_data():

    # Load your metadata
    test_metadata_df = pd.read_csv('mcv-scripted-en-v23.0/cv-corpus-23.0-2025-09-05/en/test.tsv', sep='\t')

    # Filter by accents
    accents = ['india', 'zimbabwe', 'united states', 'hong kong']
    def matches_target(accent_value):
        if pd.isna(accent_value):
            return False
        
        accent_lower = str(accent_value).lower()
        return any(keyword in accent_lower for keyword in accents)
    
    test_metadata_specific_accents_df = test_metadata_df[test_metadata_df['accents'].apply(matches_target)]


    # Create the destination directory if it doesn't exist
    dest_dir = 'filtered_test_data'
    os.makedirs(dest_dir, exist_ok=True)

    # Get the base directory where the original files are located
    base_dir = 'mcv-scripted-en-v23.0/cv-corpus-23.0-2025-09-05/en/clips'

    # Get the corresponding audio files of the filtered dataset and copy them to the destination directory
    for i, file_path in enumerate(test_metadata_specific_accents_df['path']):
        src_path = os.path.join(base_dir, file_path)
        dest_path = os.path.join(dest_dir, os.path.basename(file_path))
        print(i)
        try:
            shutil.copy2(src_path, dest_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error copying {src_path}: {str(e)}")

    
    # Ensure all audio files from the filtered metadata exist
    from pathlib import Path

    AUDIO_DIR = "filtered_test_data"

    def check_file_exists(row):
        audio_path = Path(AUDIO_DIR) / row['path']
        return audio_path.exists()

    test_metadata_specific_accents_df = test_metadata_specific_accents_df[test_metadata_specific_accents_df.apply(check_file_exists, axis=1)]
    
    # Save the filtered metadata to a new TSV file
    test_metadata_specific_accents_df.to_csv('./test_data_filtered_accents.tsv', sep='\t', index=False)

def filter_data() -> None:
    """
    Filter and process audio files or data.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory for filtered output files
    """
    

    filter_train_data()
    filter_train_data()

    
    # Add your filtering logic here
    print("Filter completed successfully!")


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 1:
        print("Usage: python filter_data.py")
        sys.exit(1)
    
    filter_data()


if __name__ == "__main__":
    main()