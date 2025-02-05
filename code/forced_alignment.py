# SPDX-FileCopyrightText: 2025 Lian Remme <lian.remme@uni-duesseldorf.de>
#
# SPDX-License-Identifier: MIT

import os
import re
from typing import List

import torch
import torchaudio
from num2words import num2words
from torchaudio.functional import TokenSpan


def compute_alignments(waveform: torch.Tensor, transcript: List[str], model: torchaudio.pipelines,
                       aligner: torchaudio.pipelines, tokenizer: torchaudio.pipelines, device: torch.device) \
        -> (torch.Tensor, List[TokenSpan]):
    """
    Computes the alignments of the given waveform and transcript using the given model, aligner and tokenizer.
    :param waveform: The waveform to be aligned.
    :param transcript: The transcript to be aligned.
    :param model: The model to be used for alignment.
    :param aligner: The aligner to be used for alignment.
    :param tokenizer: The tokenizer for the transcript.
    :param device: The device on which the alignment is done.
    :return: The tensor of the emission and the list of token spans.
    """
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans


def _score(spans: List[TokenSpan]) -> float:
    """
    Computes the average score of the given spans.
    :param spans: A list of TokenSpans, containing of token, a start frame, an end frame, and a score.
    :return: The average score of these TokeSpans.
    """
    return sum(s.score * len(s) for s in spans) / len(spans)


def normalize_uroman(text):
    """
    Normalizes the given text.
    :param text: A potentially unnormalized text.
    :return: A normalized text.
    """
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z'0-9 ])", " ", text)
    text = re.sub(' +', ' ', text)
    text.strip()
    text = text.split()
    for i in range(len(text)):
        if text[i].isnumeric():
            text[i] = num2words(text[i])
            text[i] = text[i].replace("’", "'")
            text[i] = re.sub("([^a-z' ])", " ", text[i])
            text[i] = re.sub(' +', '', text[i])
    return text


def save_result_as_csv(result_file, waveform, spans, num_frames, transcript, sample_rate):
    """
    Saves the result as csv format, in the form of "Word, Start_ms, End_ms, Score".
    :param result_file: The path to the result file.
    :param waveform: The waveform of the audio file.
    :param spans: The spans of the alignment.
    :param num_frames: The number of frames of the waveform.
    :param transcript: The transcript of the audio file.
    :param sample_rate: The sample rate of the audio file.
    """
    with open(result_file, "a") as fp:
        for i in range(len(spans)):
            print(f"Span {i}: {spans[i]}")
            ratio = waveform.size(1) / num_frames
            x0 = int(ratio * spans[i][0].start)
            x1 = int(ratio * spans[i][-1].end)
            fp.write(f"{transcript[i]}, {x0 / sample_rate}, {x1 / sample_rate}, {_score(spans[i])}\n")


def forced_alignment(audio_path: str, transcript_path: str, result_path: str, audio_format: str = '.wav'):
    """
    Applies forced alignment to all audio files in the given directory.
    Writes result in .csv format to the result path.

    :param audio_path: Path to the audio files.
    :param transcript_path: Path to the transcript files, need to be .txt files.
    :param result_path: Path to the result files.
    :param audio_format: Format of the audio files. Default is '.wav'.
    """
    audio_files = [d for d in os.listdir(audio_path) if os.path.isfile(os.path.join(audio_path, d)) and
                   d.endswith(audio_format)]

    for audio_name in audio_files:
        print("Starting to align " + audio_name + "...")
        file_name = audio_name.split('.')[0]
        audio_file = os.path.join(audio_path, audio_name)
        subtitles_file = os.path.join(transcript_path, file_name + ".txt")
        result_file = os.path.join(result_path, file_name + ".csv")
        with open(result_file, "w") as fp:
            fp.write("Word, Start_ms, End_ms, Score\n")
        with open(subtitles_file, "r") as fp:
            transcript_raw = fp.read()
        waveform, sample_rate = torchaudio.load(audio_file)
        transcript = normalize_uroman(transcript_raw)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle = torchaudio.pipelines.MMS_FA
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        model = bundle.get_model(with_star=False).to(device)

        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()

        emission, token_spans = compute_alignments(waveform, transcript, model, aligner, tokenizer, device)
        num_frames = emission.size(1)

        save_result_as_csv(result_file, waveform, token_spans, num_frames, transcript, bundle.sample_rate)


def list_to_rttm_string(file_id: str, l: List) -> str:
    """
    Converts a list of the form [start_time, end_time, speaker] to the string
    "SPEAKER {file_id} 1 {start_time:.2f} {duration:.2f} <NA> <NA> {speaker} <NA> <NA>".
    :param file_id: id of the file.
    :param l: List of the form [start_time, end_time, speaker].

    :return: "SPEAKER {file_id} 1 {start_time:.2f} {duration:.2f} <NA> <NA> {speaker} <NA> <NA>"
    """
    return f"SPEAKER {file_id} 1 {l[0]:.2f} {(l[1] - l[0]):.2f} <NA> <NA> {l[2]} <NA> <NA>"


def merge_close_numbers(l: List, threshold=0.5) -> List:
    """
    Merges the elements next to each other in a list in the form of [start_time, end_time, speaker] if gap of end_time
    of the first element and the start_time of the second one are closer than the threshold.
    Args:
    :param l: List in the form of [start_time, end_time, speaker].
    :param threshold: How close end_time and start_time have to be to be merged.
    :return: The merged list.
    """
    merged_list = [l[0]]
    for i in l[1:]:
        if i[0] - merged_list[-1][1] < threshold:
            merged_list[-1][1] = i[1]  # Overwrite end_time
        else:
            merged_list.append(i)
    return merged_list


def convert_csv_to_rttm(input_csv_path, output_rttm_path):
    """
    Saves the entries of a csv file in rttm format.
    The csv file has the form of "Word, Start_ms, End_ms, Score, Speaker". Utterances of the same speaker that are less
    than 0.5 seconds apart are merged.

    :param input_csv_path: The csv file of the form "Word, Start_ms, End_ms, Score, Speaker".
    :param output_rttm_path: The rttm file which rsulted from the csv file.
    """
    with open(input_csv_path, 'r') as fp:
        input = fp.read()
    file_id = os.path.basename(input_csv_path).split('.')[0]
    words = input.split('\n')[1:]
    words = [w for w in words if w != ""]
    result = {}
    for w in words:
        _, start_time, end_time, _, speaker = w.split(', ')
        start_time = float(start_time)
        end_time = float(end_time)
        if speaker in result:
            result[speaker].append([start_time, end_time, speaker])
        else:
            result[speaker] = [[start_time, end_time, speaker]]
    for speaker in result.keys():
        result[speaker] = merge_close_numbers(result[speaker])
    result = [item for sublist in list(result.values()) for item in sublist]
    result = sorted(result, key=lambda r: r[0])
    rttm_str = [list_to_rttm_string(file_id, r) for r in result]
    rttm_str = '\n'.join(rttm_str)
    with open(output_rttm_path, 'w') as fp:
        fp.write(rttm_str)