# ============================================================================
# IMPORTS AND INITIALIZATION
# ============================================================================
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
import torch
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

def finetune_with_lora():
    # ========================================================================
    # DEVICE AND MODEL SETUP
    # ========================================================================
    print(torch.__version__)
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()

    # ========================================================================
    # LOAD AND FILTER TRAINING DATA
    # ========================================================================
    file_path = 'train_data_filtered_accents.tsv'
    train_metadata_df = pd.read_csv(file_path, sep='\t', low_memory=False)
    train_metadata_df.to_csv('train_metadata.csv', index=False)

    file_path = 'test_data_filtered_accents.tsv'
    test_metadata_df = pd.read_csv(file_path, sep='\t', low_memory=False)
    test_metadata_df.to_csv('test_metadata.csv', index=False)

    AUDIO_DIR = "filtered_train_data"
    def check_file_exists(row):
        audio_path = Path(AUDIO_DIR) / row['path']
        return audio_path.exists()

    train_metadata_filtered_df = train_metadata_df[train_metadata_df.apply(check_file_exists, axis=1)]
    print(f"Original train dataframe shape: {train_metadata_df.shape}")
    print(f"Filtered train dataframe shape: {train_metadata_filtered_df.shape}")
    train_metadata_filtered_df['accents'].value_counts()

    # ========================================================================
    # LOAD AND FILTER TEST DATA
    # ========================================================================
    AUDIO_DIR = "filtered_test_data"
    def check_file_exists(row):
        audio_path = Path(AUDIO_DIR) / row['path']
        return audio_path.exists()

    test_metadata_filtered_df = test_metadata_df[test_metadata_df.apply(check_file_exists, axis=1)]
    print(f"Original test dataframe shape: {test_metadata_df.shape}")
    print(f"Filtered test dataframe shape: {test_metadata_filtered_df.shape}")
    test_metadata_filtered_df['accents'].value_counts()

    # ========================================================================
    # SPLIT DATA BY ACCENT
    # ========================================================================
    US_train_metadata_df = train_metadata_filtered_df[train_metadata_filtered_df['accents'].str.contains('united states', case=False)]
    US_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('united states', case=False)]

    indian_train_metadata_df = train_metadata_filtered_df[train_metadata_filtered_df['accents'].str.contains('india', case=False, na=False)]
    indian_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('india', case=False, na=False)]

    hongkong_train_metadata_df = train_metadata_filtered_df[train_metadata_filtered_df['accents'].str.contains('hong kong', case=False, na=False)]
    hongkong_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('hong kong', case=False, na=False)]

    zimbabwe_train_metadata_df = train_metadata_filtered_df[train_metadata_filtered_df['accents'].str.contains('zimbabwe', case=False, na=False)]
    zimbabwe_test_metadata_df = test_metadata_filtered_df[test_metadata_filtered_df['accents'].str.contains('zimbabwe', case=False, na=False)]

    # ========================================================================
    # AUDIO PROCESSING FUNCTION
    # ========================================================================
    def transcribe_file(audio_path, index=-1, model=None, device=None, display_audio=False):
        """Load, resample, and convert audio to mono format."""
        waveform, sample_rate = torchaudio.load(audio_path)

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

    # ========================================================================
    # INFERENCE FUNCTION - TEST MODEL ON ACCENTED AUDIO
    # ========================================================================
    AUDIO_DIR = "filtered_test_data"
    BATCH_SIZE = 3

    def test_accented_audios(test_metadata_df):
        """Evaluate model performance on test set, compute WER scores."""
        all_wer_scores = []
        all_transcriptions = []
        all_info = []

        total_samples = len(test_metadata_df)

        for start_idx in range(0, total_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_samples)
            batch_df = test_metadata_df.iloc[start_idx:end_idx]

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
            padded_waveforms = padded_waveforms.to(device)

            # Run inference
            with torch.no_grad():
                logits = model(padded_waveforms).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)

            # Calculate WER for each sample
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

    # ========================================================================
    # BALANCE TRAINING DATA BY ACCENT
    # ========================================================================
    max_count = max(len(indian_train_metadata_df), len(hongkong_train_metadata_df), len(zimbabwe_train_metadata_df))
    US_train_metadata_df = US_train_metadata_df[:max_count]

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

    # ========================================================================
    # CREATE WEIGHTED SAMPLER FOR BALANCED TRAINING
    # ========================================================================
    sample_weights = []
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
        """Custom dataloader with weighted sampling."""
        return DataLoader(
            trainer_self.train_dataset,
            batch_size=trainer_self.args.train_batch_size,
            sampler=sampler,
            collate_fn=trainer_self.data_collator,
            drop_last=False,
            num_workers=4,
            pin_memory=True
        )

    # ========================================================================
    # PREPARE DATASETS
    # ========================================================================
    train_df.shape
    train_paths_in_filtered = train_df['path'].isin(train_metadata_filtered_df['path'])
    train_df = train_df[train_paths_in_filtered]
    train_df.shape

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(test_df)

    def prepare_dataset(batch, dataset_type = 'train'):
        """Load and preprocess audio and text for training."""
        audio_path = str(Path(f'filtered_{dataset_type}_data') / batch['path'])
        speech_array, sampling_rate = torchaudio.load(audio_path)

        # Resample to 16kHz
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = resampler(speech_array)

        # Convert to mono
        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        speech_array = speech_array.squeeze().numpy()

        # Extract features
        batch["input_values"] = processor.feature_extractor(
            speech_array,
            sampling_rate=16000
        ).input_values[0]

        # Tokenize labels
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

    # ========================================================================
    # CUSTOM DATA COLLATOR FOR PADDING
    # ========================================================================
    class DataCollatorCTCWithPadding:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, features):
            """Pad input and label sequences to same length within batch."""
            input_features = [torch.tensor(feature["input_values"]) for feature in features]
            label_features = [torch.tensor(feature["labels"]) for feature in features]

            input_features_padded = pad_sequence(
                input_features,
                batch_first=True,
                padding_value=self.processor.feature_extractor.padding_value
            )
            labels_padded = pad_sequence(
                label_features,
                batch_first=True,
                padding_value=-100
            )

            output = {
                "input_values": input_features_padded,
                "labels": labels_padded
            }

            return output

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # ========================================================================
    # CONFIGURE LORA (LOW-RANK ADAPTATION) FOR FINE-TUNING
    # ========================================================================
    model.train()
    model.freeze_feature_encoder()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ========================================================================
    # TRAINING SETUP AND EXECUTION
    # ========================================================================
    learning_rate = 3e-5
    print(f"Training with learning rate: {learning_rate}")
    model.train()

    Trainer.get_train_dataloader = get_train_dataloader

    training_args = TrainingArguments(
        output_dir=f"/scratch/kayastha.r/temp_lr{learning_rate}",
        group_by_length=False,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=200,
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

    trainer.train()

    # ========================================================================
    # EVALUATION AND TESTING
    # ========================================================================
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    best_epoch = eval_losses.index(min(eval_losses)) + 1
    print(f"Best epoch: {best_epoch}, Best eval loss: {min(eval_losses)}")

    print("Training complete!")

    model.eval()
    metrics = trainer.evaluate()

    # Test on each accent group
    test_accented_audios(indian_test_metadata_df)
    test_accented_audios(hongkong_test_metadata_df)
    test_accented_audios(zimbabwe_test_metadata_df)
    test_accented_audios(US_test_metadata_df)
