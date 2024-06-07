# Playing with Voices: Tabletop Role-Playing Game Recordings as a Diarization Challenge

![Python Version 3.11.6](https://img.shields.io/badge/Python-3.11.6-green)


Supplementary material for the submission to ACL.

## Abstract

> This paper provides a proof of concept that audio of tabletop role-playing games (TTRPG) could serve as a new challenge for diarization systems. TTRPGs are games that are carried out mostly by conversation. Participants often alter their voices to indicate that they are talking as a fictional character in a game. Audio processing systems are susceptible to voice conversion with or without technological assistance. TTRPG presents a conversational phenomenon in which voice conversion is an inherent characteristic for an immersive gaming experience. We present the creation of a small TTRPG audio dataset and compare it against the AMI and the ICSI corpus. The performance of two diarizers, pyannote.audio and wespeaker, were evaluated. We observed that TTRPGs' properties result in a higher confusion rate for both diarizers.
Additionally, wespeaker strongly underestimates the number of speakers in the TTRPG audio files.
We propose TTRPG audio as a promising challenge for diarization systems.

Required python packages and their versions can be found in `requirements.txt`.

## Contents

In this repository, we provide the following:

- [Code: How the diarizer was applied](#Application-of-the-diarizer)
- [Code: How we applied forced alignment](#Application-of-forced-alignment)
- [Code: How we converted the forced alignment results to rttm](#Forces-Alignment-csv-to-rttm)
- [Code: How the evaluation has been done](#evaluation-of-diarization)
- [Code: How we calculated the amount of overlapping speech](#calculation-of-overlapping-speech)
- [Code: How we calculated the amount of interjections](#calculation-of-interjections)
- [Code: How we applied the Mann-Whitney U test](#application-of-mann-whitney-u-test)
- [Links: The YouTube videos we used for our TTRPG dataset](#links-to-youtube-videos)


## Application of the diarizer 

Calling the diarizer as we did in our work. The diarizer can be called by

```console
python main.py -d <options>
```
The available options are:
- `-a`   (or) `--audio_path`: Path to the audio files that should be diarized. **Required.**
- `-r`   (or) `--result_path`: Path to the directory where the results should be saved. The files will have the same file name as their respective audio files. **Required.**
- `-t`   (or) `--access_token`: The hugging face access token. You need to have access [pyannote.audio 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1). **Required.**
- `-f`   (or) `--audio_format`: The format of the audio files. The default is `wav`.
- `-w`   (or) `--reference_path`: Path to the reference files. The reference files need to be `rttm` files and have the same file name as their respective audio files. Only required if `c` is set (see below).
- `-c`  (or) `--consider_speaker_no`: If set, the diarizer will receive the number of expected speakers as argument. The number is taken from the reference files.

## Application of forced alignment

Applying forced alignment as we did in our work.

```console
python main.py -fa <options>
```

The results of the forced alignment will be saved in `csv` files containing start and end time of each word and the confidence of the alignment (`Word, Start_ms, End_ms, Score`).

The available options are:
- `-a`   (or) `--audio_path`: Path to the audio files that should be aligned. **Required.**
- `-r`   (or) `--result_path`: Path to the directory where the results should be saved. The files will have the same file name as their respective audio files. **Required.**
- `-w`   (or) `--reference_path`: Path to the transcript files. The reference files need to be `txt` files and have the same file name as their respective audio files. **Required.**
- `-f`   (or) `--audio_format`: The format of the audio files. The default is `wav`.

## Forces Alignment csv to rttm

Converting `csv` files in the form of `Word, Start_ms, End_ms, Score, Speaker` into an `rttm` file.

```console
python main.py -c2r <options>
```
Entries of the same speaker that are less than 500ms apart are merged into one entry.

The available options are:
- `-r`   (or) `--result_path`: Path to the `rttm` result file **Required.**
- `-w`   (or) `--reference_path`: Path to the `csv` reference file. **Required.**


## Evaluation of diarization

Evaluating the diarization by calculation diarization error rate (DER), confusion, false alarm, and missed detection.
Additionally, the `csv` result file will contain the number of detected speakers, the actual number of speakers, the speaker ratio, and the length of the audio file(s).

```console
python main.py -e <options>
```

The available options are:
- `-a`   (or) `--audio_path`: Path to the audio files. **Required.**
- `-r`   (or) `--result_path`: Path to the hypothesis files. The reference files need to be `rttm` files and have the same file name as their respective audio files. **Required.**
- `-w`   (or) `--reference_path`: Path to the reference/ground-truth files. The reference files need to be `rttm` files and have the same file name as their respective audio files. **Required.**
- `-e`   (or) `--evaluate_file`: The `csv` file for the evaluation results. **Required.**
- `-f`   (or) `--audio_format`: The format of the audio files. The default is `wav`.

## Calculation of overlapping speech

Calculates the amount of overlapping speech in the `rttm` files.
Writes the results to a `csv` file.

```console
python main.py -o <options>
```

The available options are:
- `-w`   (or) `--reference_path`: Path to the reference/ground-truth files. The reference files need to be `rttm` files. **Required.**
- `-e`   (or) `--evaluate_file`: The `csv` file for the evaluation results. **Required.**

## Calculation of interjections

Calculates the amount of filler words in the `txt` files.
Writes the results to a `csv` file.

```console
python main.py -fw <options>
```

The available options are:
- `-w`   (or) `--reference_path`: Path to the reference files. The reference files need to be `txt` files. **Required.**
- `-e`   (or) `--evaluate_file`: The `csv` file for the evaluation results. **Required.**

Before the calculation can be done, one has to call
```console
python -m spacy download en_core_web_trf
```

## Application of Mann-Whitney U test

Calculates the Mann-Whitney U test for two datasets in the two given files.
It is assumed that the two given files contain one datapoint as float in each line.
The result (Mann-Whitney U statistic, the p-value) will be printed to the console.

```console
python main.py -mw <options>
```

- `-x`   (or) `--dataset_x`: Path to the file that contains the first dataset. **Required.**
- `-y`   (or) `--dataset_y`: Path to the file that contains the second dataset. **Required.**

## Links to YouTube videos

The links to the used YouTube videos can be found in the `links.txt` file.


## Disclosure about AI usage

GitHub Copilot has been used to assist during code-writing. 
Copilot helped writing some documentation strings, deciding on variable and function names and wrote first drafts of some functions.