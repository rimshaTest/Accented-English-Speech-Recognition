#!/bin/bash


#SBATCH --job-name=ASP_download_dataset

#SBATCH --output=output_ASP_download_dataset.txt

#SBATCH --error=error_ASP_download_dataset.txt

#SBATCH --mail-user=kayastha.r@northeastern.edu

#SBATCH --mail-type=ALL

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

mkdir -p logs


module load anaconda3/2024.06

# Download from Google Drive
echo "Downloading data from Google Drive..."

# Install gdown if not already installed
pip install gdown

# Download from Google Drive using gdown
echo "Downloading data..."

# Get shareable links from Google Drive, extract FILE_IDs
gdown "https://drive.google.com/file/d/1INsvK-7-DbHeKyfG9_JCwr02gnYbtVjy/view?usp=sharing" -O filtered_train_data.zip
gdown "https://drive.google.com/file/d/1XAT-TylabtT1MOFc1c3LkfDon0X2HOU5/view?usp=sharing" -O filtered_test_data.zip
gdown "https://drive.google.com/file/d/1nPP7OmILxoA3xdnVpzTiwxkiVl1kCUrR/view?usp=sharing" -O train_data_filtered_accents.tsv
gdown "https://drive.google.com/file/d/1AVW0ZMSScUyPAs5P7SD_EJbUKdWHhcIV/view?usp=sharing" -O test_data_filtered_accents.tsv

# Verify downloads
echo "Checking file sizes..."
ls -lh *.zip *.tsv

# Unzip
unzip -q filtered_train_data.zip
unzip -q filtered_test_data.zip


