
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
from datasets import load_dataset, Audio
import pandas as pd
from collections import Counter
import json
import argparse
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from jiwer import wer
import shutil
import zipfile
from IPython.display import Audio, display
import torchaudio

class baseline_evaluation:

    def __init__(self):
        print(torch.__version__)

        torch.random.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.model = None
        self.processor = None
        self.test_metadata_filtered_df = None
        self.US_test_metadata_df = None
        self.indian_test_metadata_df = None
        self.hongkong_test_metadata_df = None
        self.zimbabwe_test_metadata_df = None
        print(self.device)

    def prepare_data(self):
        file_path = 'test_data_filtered_accents.tsv'

        test_metadata_df = pd.read_csv(file_path, sep='\t')
        test_metadata_df.head(5)

        test_metadata_df.to_csv('test_metadata.csv', index=False)

        # Extract test data, skipping files that fail
        if os.path.exists('filtered_test_data.zip'):
            print("Extracting test data...")
            with zipfile.ZipFile('filtered_test_data.zip', 'r') as zip_ref:
                for member in zip_ref.namelist():
                    try:
                        zip_ref.extract(member, '.')
                    except (PermissionError, zipfile.BadZipFile) as e:
                        print(f"Skipping {member}: {e}")
                        continue
            print("Test data extraction complete (with skips)")


        AUDIO_DIR = "filtered_test_data"  # Assuming the audio files are in this directory

        def check_file_exists(row):
            audio_path = Path(AUDIO_DIR) / row['path']
            return audio_path.exists()

        test_metadata_filtered_df = test_metadata_df[test_metadata_df.apply(check_file_exists, axis=1)]
        self.test_metadata_filtered_df = test_metadata_filtered_df
        print(f"Original test dataframe shape: {test_metadata_df.shape}")
        print(f"Filtered test dataframe shape: {test_metadata_filtered_df.shape}")

        self.US_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('united states', case=False)]

        self.indian_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('india', case=False, na=False)]
        
        self.hongkong_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('hong kong', case=False, na=False)]

        self.zimbabwe_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('zimbabwe', case=False, na=False)]

    def run_evaluation(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.test_accented_audios(self.indian_test_metadata_df)

        self.test_accented_audios(self.hongkong_test_metadata_df)

        self.test_accented_audios(self.zimbabwe_test_metadata_df)

        self.test_accented_audios(self.US_test_metadata_df)

    def transcribe_file(self, audio_path, index=-1, model=None, device=None, display_audio=False):
        waveform, sample_rate = torchaudio.load(audio_path)

        # Only display if requested and index is 0
        if display_audio and index == 0:
            audio_widget = Audio(audio_path)
            display(audio_widget)

        # Resample to 16kHz
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate
            )
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform

    def test_accented_audios(self, test_metadata_df):
        AUDIO_DIR = "filtered_test_data"
        BATCH_SIZE = 3
        all_wer_scores = []
        all_transcriptions = []
        all_info = []

        # Loop through dataset in batches
        total_samples = len(test_metadata_df)

        for start_idx in range(0, total_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_samples)
            batch_df = test_metadata_df.iloc[start_idx:end_idx]  # ← Get this batch

            batch_waveforms = []
            batch_info = []

            # Load audio for this batch
            for idx, row in batch_df.iterrows():
                audio_path = Path(AUDIO_DIR) / row['path']

                if audio_path.exists():
                    waveform = self.transcribe_file(str(audio_path), idx, self.model, self.device)

                    waveform = waveform.squeeze(0)

                    batch_waveforms.append(waveform)

                    batch_info.append({
                        'accent': row['accents'],
                        'ground_truth': row['sentence']
                    })

            padded_waveforms = pad_sequence(batch_waveforms, batch_first=True, padding_value=0.0)

            # Move to device
            padded_waveforms = padded_waveforms.to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(padded_waveforms).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = self.processor.batch_decode(predicted_ids)

            for i in range(len(batch_waveforms)):
                wer_score = wer(batch_info[i]['ground_truth'].lower(), transcriptions[i].lower())
                all_wer_scores.append(wer_score)
                all_transcriptions.append(transcriptions[i])
                all_info.append(batch_info[i])

                if start_idx==0:
                    print(f"\nSample {i}:")
                    print(f"  Ground truth: {batch_info[i]['ground_truth']}")
                    print(f"  Prediction:   {transcriptions[i]}")
                    print(f"  WER:          {wer_score * 100:.2f}%")

            del padded_waveforms, logits, predicted_ids
            torch.cuda.empty_cache()

        avg_wer = sum(all_wer_scores) / len(all_wer_scores)

        print(f"Average WER: {avg_wer * 100:.2f}%")
