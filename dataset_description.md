# Data Description

> This dataset contains 2,000 environmental audio recordings, 
> perfectly balanced across 50 distinct classes (40 clips per class). 
> All audio files are 5 seconds long, single-channel (mono), and sampled at 44.1 kHz.

> To keep the downloads organized,
> the audio files and their corresponding metadata CSVs have been bundled
> together into two main archives: one for training and one for testing.

# Files and Directories

## train_audio.zip

> Contains the **training .wav** audio files.
> Includes **train.csv** : The metadata file containing the filename, 
> the numeric target ID (0 to 49), and the human-readable category name.
> You will use this to train your model.

## test_audio.zip

> Contains the test .wav audio files.
> Includes **test.csv**: Contains only the filename column.
> These are the files you need to run your model on to predict the target labels.

## sample_submission.csv:

> A perfectly formatted example of how your final submission file should look. 
> It contains the filename and a placeholder target column.


# DATA COLUMNS 
>Inside the CSV files, you will encounter the following fields:

## filename: The exact name of the .wav file (e.g., 1-100032-A-0.wav).

## target: An integer ranging from 0 to 49 representing the acoustic class. 
   ## (This is what you need to predict for the test set!)

## category: A string representing the semantic label of the sound (e.g., dog, rain, crying_baby). This is provided in the training data to help you understand what the numeric target represents.