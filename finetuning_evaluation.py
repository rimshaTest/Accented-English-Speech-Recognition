import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import Audio
import pandas as pd
from pathlib import Path
from IPython.display import Audio, display
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from jiwer import wer
from torch.utils.data import WeightedRandomSampler, DataLoader
from datasets import Dataset
from transformers import TrainingArguments, Trainer

def finetune_common_voice():
    """
    Fine-tune Wav2Vec2 model on accented English speech data from multiple regions.
    Performs weighted sampling by accent, trains the model, and evaluates on test sets.
    """
    print(torch.__version__)

    # Set random seed and device configuration
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    # Load pre-trained processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()

    # ===== Load and filter training metadata =====
    file_path = 'train_data_filtered_accents.tsv'

    train_metadata_df = pd.read_csv(file_path, sep='\t', low_memory=False)

    train_metadata_df.to_csv('train_metadata.csv', index=False)

    file_path = 'test_data_filtered_accents.tsv'

    test_metadata_df = pd.read_csv(file_path, sep='\t', low_memory=False)

    test_metadata_df.to_csv('test_metadata.csv', index=False)

    # ===== Filter training audio files =====
    AUDIO_DIR = "filtered_train_data" 

    def check_file_exists(row):
        """Check if audio file exists at the given path."""
        audio_path = Path(AUDIO_DIR) / row['path']
        return audio_path.exists()

    train_metadata_filtered_df = train_metadata_df[train_metadata_df.apply(check_file_exists, axis=1)]

    train_metadata_filtered_df['accents'].value_counts()


    # ===== Filter test audio files =====
    AUDIO_DIR = "filtered_test_data"

    def check_file_exists(row):
        """Check if audio file exists at the given path."""
        audio_path = Path(AUDIO_DIR) / row['path']
        return audio_path.exists()

    test_metadata_filtered_df = test_metadata_df[test_metadata_df.apply(check_file_exists, axis=1)]

    test_metadata_filtered_df['accents'].value_counts()

    # ===== Separate data by accent =====
    US_train_metadata_df = train_metadata_filtered_df[train_metadata_filtered_df['accents'].str.contains('united states', case=False)]

    US_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('united states', case=False)]

    indian_train_metadata_df = train_metadata_filtered_df[train_metadata_filtered_df['accents'].str.contains('india', case=False, na=False)]

    indian_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('india', case=False, na=False)]

    hongkong_train_metadata_df = train_metadata_filtered_df[train_metadata_filtered_df['accents'].str.contains('hong kong', case=False, na=False)]

    hongkong_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('hong kong', case=False, na=False)]

    zimbabwe_train_metadata_df = train_metadata_filtered_df[train_metadata_filtered_df['accents'].str.contains('zimbabwe', case=False, na=False)]

    zimbabwe_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('zimbabwe', case=False, na=False)]


    def transcribe_file(audio_path, index=-1, model=None, device=None, display_audio=False):
        """
        Load and preprocess audio file: resample to 16kHz and convert to mono.
        
        Args:
            audio_path: Path to audio file
            index: Sample index for display purposes
            model: Model (unused, kept for compatibility)
            device: Device (unused, kept for compatibility)
            display_audio: Whether to display audio widget
            
        Returns:
            Preprocessed waveform tensor
        """
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


    # ===== Evaluation configuration =====
    AUDIO_DIR = "filtered_test_data"
    BATCH_SIZE = 3

    def test_accented_audios(test_metadata_df):
        """
        Evaluate model on accented audio samples and compute WER (Word Error Rate).
        Processes data in batches for efficiency.
        
        Args:
            test_metadata_df: DataFrame with test samples (path, accents, sentence)
        """
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
                    waveform = transcribe_file(str(audio_path), idx, model, device)

                    waveform = waveform.squeeze(0)

                    batch_waveforms.append(waveform)

                    batch_info.append({
                        'accent': row['accents'],
                        'ground_truth': row['sentence']
                    })

            padded_waveforms = pad_sequence(batch_waveforms, batch_first=True, padding_value=0.0)

            # Move to device
            padded_waveforms = padded_waveforms.to(device)

            # Inference
            with torch.no_grad():
                logits = model(padded_waveforms).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)

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

    # ===== Prepare weighted sampling by accent =====
    max_count = max(len(indian_train_metadata_df), len(hongkong_train_metadata_df), len(zimbabwe_train_metadata_df))
    US_train_metadata_df = US_train_metadata_df[:max_count]

    # Count samples per accent
    sizes = {
        "indian": len(indian_train_metadata_df),
        "hongkong": len(hongkong_train_metadata_df),
        "zimbabwe": len(zimbabwe_train_metadata_df),
        "us": len(US_train_metadata_df),
    }
    print(sizes)

    weights_by_accent = {k: 1.0/v for k, v in sizes.items()}

    train_df = pd.concat([
        indian_train_metadata_df,
        hongkong_train_metadata_df,
        zimbabwe_train_metadata_df,
        US_train_metadata_df
    ])

    test_df = pd.concat([
        indian_test_metadata_df,
        hongkong_test_metadata_df,
        zimbabwe_test_metadata_df,
        US_test_metadata_df
    ])


    sample_weights = []

    # Accents are concatenated in this order:
    blocks = [
        ("indian", len(indian_train_metadata_df)),
        ("hongkong", len(hongkong_train_metadata_df)),
        ("zimbabwe", len(zimbabwe_train_metadata_df)),
        ("us", len(US_train_metadata_df)),
    ]

    for accent, count in blocks:
        w = weights_by_accent[accent]
        sample_weights.extend([w] * count)

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


    def get_train_dataloader(trainer_self):
        return DataLoader(
            trainer_self.train_dataset,
            batch_size=trainer_self.args.train_batch_size,
            sampler=sampler,
            collate_fn=trainer_self.data_collator,
            drop_last=False,
            num_workers=4,
            pin_memory=True
        )


    train_paths_in_filtered = train_df['path'].isin(train_metadata_filtered_df['path'])
    train_df = train_df[train_paths_in_filtered]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)


    # Convert pandas DataFrame to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(test_df)

    def prepare_dataset(batch, dataset_type = 'train'):
        # Construct full audio path as STRING
        audio_path = str(Path(f'filtered_{dataset_type}_data') / batch['path'])

        # Load audio
        speech_array, sampling_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = resampler(speech_array)

        # Convert to mono
        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        # Convert to numpy array (required by processor)
        speech_array = speech_array.squeeze().numpy()

        # Process audio
        batch["input_values"] = processor.feature_extractor(
            speech_array,
            sampling_rate=16000
        ).input_values[0]

        # Process labels
        text = batch["sentence"].upper()
        batch["labels"] = processor.tokenizer.encode(text)

        return batch

    # Process datasets
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset(batch, 'train'),
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False
    )

    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset(batch, 'test'),
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False
    )


    class DataCollatorCTCWithPadding:
        """Custom data collator for CTC loss with dynamic padding of variable-length sequences."""
        
        def __init__(self, processor):
            """Initialize with Wav2Vec2 processor."""
            self.processor = processor

        def __call__(self, features):
            """
            Collate batch of samples with padding.
            
            Args:
                features: List of feature dictionaries with 'input_values' and 'labels'
                
            Returns:
                Dictionary with padded input_values and labels
            """
            # Extract tensors
            input_features = [torch.tensor(feature["input_values"]) for feature in features]
            label_features = [torch.tensor(feature["labels"]) for feature in features]

            # Pad input features and labels to same length
            input_features_padded = pad_sequence(
                input_features,
                batch_first=True,
                padding_value=self.processor.feature_extractor.padding_value
            )
            labels_padded = pad_sequence(
                label_features,
                batch_first=True,
                padding_value=-100  # Ignore padding tokens in CTC loss
            )

            # Create output batch
            output = {
                "input_values": input_features_padded,
                "labels": labels_padded
            }

            return output


    # ===== Prepare trainer and training configuration =====
    
    # Check if model is in training mode
    print(f"Model training mode: {model.training}")

    # Verify dataset preparation
    sample = train_dataset[0]
    decoded = processor.tokenizer.decode(sample['labels'])
    print("Sample decoded:", decoded)

    # Initialize data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Set hyperparameters
    learning_rate = 9e-5

    print(f"Training with learning rate: {learning_rate}")
    model.train()

    # Override default dataloader with weighted sampler
    Trainer.get_train_dataloader = get_train_dataloader

    # Configure training arguments
    training_args = TrainingArguments(
    output_dir=f"/scratch/kayastha.r/temp_lr{learning_rate}",
    group_by_length=False,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="no", 
    num_train_epochs=2,
    fp16=True,
    logging_steps=100,
    learning_rate=learning_rate,
    remove_unused_columns=False,
    report_to="none",
    load_best_model_at_end=False,
    max_grad_norm=1.0,
    dataloader_num_workers=8,
    )

    trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    )

    # Start fine-tuning
    trainer.train()

    # ===== Evaluate model =====
    
    # Find best epoch from training logs
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    best_epoch = eval_losses.index(min(eval_losses)) + 1
    print(f"Best epoch: {best_epoch}, Best eval loss: {min(eval_losses)}")

    print("Training complete!")

    # Set model to evaluation mode
    model.eval()

    # Run evaluation on validation set
    trainer.evaluate()

    # ===== Test on accented speech by region =====
    print("\n=== Evaluating Indian accent ===")
    test_accented_audios(indian_test_metadata_df)

    print("\n=== Evaluating Hong Kong accent ===")
    test_accented_audios(hongkong_test_metadata_df)

    print("\n=== Evaluating Zimbabwe accent ===")
    test_accented_audios(zimbabwe_test_metadata_df)

    print("\n=== Evaluating US accent ===")
    test_accented_audios(US_test_metadata_df)