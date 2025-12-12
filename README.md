# Accented-English-Speech-Recognition

This project evaluates and adapts the Wav2vec 2.0 Automatic Speech Recognition (ASR) model for English with non-native accents. It benchmarks the model's performance degradation on Indian, Hong Kong, and Zimbabwean English from the Common Voice dataset and compares two adaptation strategies: full fine-tuning and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA).

# Setup

Note: This project was implemented heavily through the cluster and Colab Pro due to the extensive amount of computational resources it requires. Hence, the same is recommended for running this.

## Cloning the repo
```
git clone https://github.com/rimshaTest/Accented-English-Speech-Recognition
cd Accented-English-Speech-Recognition
```

## Installing the required dependencies

You can install the required libraries with the following commands:

```
pip install -r requirements.txt
```

# Downloading Dataset

The dataset for this project has been sourced from Mozilla Common Voice dataset, which includes English recordings with labeled accents. It was then filtered by US accents and three other accents picked for its significant difference from the standard English: Indian, Hong Kong, and Zimbabwean.

## Downloading and Filtering Common Voice

If that does not work, you can download the dataset directly from [here](https://datacollective.mozillafoundation.org/datasets/cmflnuzw52mzok78yz6woemc1#user-content-fn-1). Please note that this downloads the entire dataset and might take time. After you unzip it into the project root, you can then run the filter_data.py script to filter the data by the four accents.

## Quick Download

You can download the filtered dataset using the download_dataset.sh script with the following commands:\

```
chmod +x download_dataset.sh
./download_dataset.sh
```

# Running Experiments

This project uses the facebook/wav2vec2-large-960h model checkpoint. The experiments are:

- b: Evaluate the baseline, pre-trained model (zero-shot).
- f: Full fine-tuning on the accented data, then evaluation.
- l: LoRA fine-tuning on the accented data, then evaluation.

You can run an experiment with the corresponding letter to main.py by running the following commands:\
    ```
    python main.py b
    ```
    \
    ```
    python main.py f
    ```
    \
    ```
    python main.py l
    ```

You will be able to see the results in the terminal output.

# License

This project is licensed under the MIT License - see the LICENSE file for details.
