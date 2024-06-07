import os
import spacy
from typing import List

import soundfile as sf

from scipy.stats import mannwhitneyu
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

diarization_metric = DiarizationErrorRate(collar=0.5, skip_overlap=False)


def count_speaker_from_rttm(rttm_file: str) -> int:
    """
    Counts the number of speakers in the rttm file.

    :param rttm_file: Path to the rttm file.
    :return: The number of speakers in the rttm file.
    """
    with open(rttm_file, 'r') as fp:
        lines = fp.read().split('\n')
    lines = list(filter(lambda s: len(s) != 0, lines))
    speakers = set()
    for line in lines:
        speakers.add(line.split(' ')[7])
    return len(speakers)


def determine_audio_length(audio_file: str) -> float:
    """
    Determines the length of the audio file in seconds.

    :param audio_file: Path to the audio file.
    :return: The length of the audio file in seconds.
    """
    audio, sample_rate = sf.read(audio_file)
    return len(audio) / sample_rate


def compare_speaker_amount_of_two_files(rttm_file_a: str, rttm_file_b: str) -> (int, int, float):
    """
    Compares the number of speakers in two rttm files.

    :param rttm_file_a: Path to the first rttm file.
    :param rttm_file_b: Path to the second rttm file.
    :return: The number of speakers in the first rttm file, the number of speakers in the second rttm file and the
    relative difference (first speaker no. / second speaker no.).
    """
    speakers_a = count_speaker_from_rttm(rttm_file_a)
    speakers_b = count_speaker_from_rttm(rttm_file_b)
    return speakers_a, speakers_b, speakers_b / speakers_a


def diarization_error(reference_file: str, hypothesis_file: str) -> (float, float, float, float):
    """
    Evaluates specifics about the diarization error for the given files.

    :param reference_file: Reference file, needs to be a rttm file.
    :param hypothesis_file: Hypothesis file, needs to be a rttm file.
    :return: The diarization error rate, confusion error rate, false alarm rate and the missed detection rate.
    """
    _, groundtruth = load_rttm(reference_file).popitem()
    _, hypothesis = load_rttm(hypothesis_file).popitem()
    errors = diarization_metric(groundtruth, hypothesis, detailed=True)
    false_alarm = errors['false alarm']
    missed_detection = errors['missed detection']
    confusion = errors['confusion']
    total = errors['total']
    false_alarm_rate = false_alarm / total
    missed_detection_rate = missed_detection / total
    confusion_rate = confusion / total
    der = false_alarm_rate + missed_detection_rate + confusion_rate
    return der, confusion_rate, false_alarm_rate, missed_detection_rate


def evaluate_all_files_diarization(reference_path: str, hypothesis_path: str, audio_path: str, result_file: str,
                                   audio_format: str = '.wav'):
    """
    Evaluates diarization error in all files in the given directories and writes the result to a csv file.
    Files in reference path are compared to files with the same name in hypothesis path.

    :param reference_path: Path to the reference directory.
    :param hypothesis_path: Path to the hypothesis directory.
    :param audio_path: Path to the audio files.
    :param result_file: File to write the result to.
    :param audio_format: Format of the audio files. Default is '.wav'.
    """
    with open(result_file, 'w') as fp:
        fp.write(f"file, audio_length_s, speaker_number_reference, speaker_number_hypothesis, "
                 f"speaker_no_relative_difference, diarization_error, confusion, false_alarm, missed_detection\n")

    files = [f for f in os.listdir(reference_path) if os.path.isfile(os.path.join(reference_path, f)) and
             f.endswith('.rttm')]
    for file in files:
        file_id = file.split('.')[0]
        audio_file = os.path.join(audio_path, file_id + audio_format)
        rttm_file_ref = os.path.join(reference_path, file)
        rttm_file_hyp = os.path.join(hypothesis_path, file)
        if os.path.isfile(rttm_file_hyp) and os.path.isfile(audio_file):
            audio_length = determine_audio_length(audio_file)
            speakers_ref, speakers_hyp, speaker_no_relative = compare_speaker_amount_of_two_files(rttm_file_ref,
                                                                                                  rttm_file_hyp)
            der, confusion, false, missed = diarization_error(rttm_file_ref, rttm_file_hyp)
            with open(result_file, 'a') as fp:
                fp.write(f"{file_id}, {audio_length}, {speakers_ref}, {speakers_hyp}, {speaker_no_relative}, {der}, "
                         f"{confusion}, {false}, {missed}\n")


def parse_rttm_file(file_path: str) -> List:
    """
    Parse an RTTM file and return a list of tuples containing speaker information.
    Each tuple contains (start_time, duration, speaker_id).
    :param file_path: Path to the rttm file.
    :return: List of tuples containing speaker information (start_time, duration, speaker_id).
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    speaker_info = []
    for line in lines:
        parts = line.strip().split()
        start_time = float(parts[3])
        duration = float(parts[4])
        speaker_id = parts[7]
        speaker_info.append((start_time, duration, speaker_id))

    return speaker_info


def calculate_overlap_duration(rttm_file_path: str):
    """
    Calculate the total duration of overlapping speech in milliseconds and the relative amount.
    :param rttm_file_path: Path to the rttm file.
    :return: The total duration of overlapping speech in milliseconds and the relative amount.
    """
    speaker_info = parse_rttm_file(rttm_file_path)
    overlap_tuple = []
    for i in range(len(speaker_info) - 1):
        j = i + 1
        start_i, duration_i, _ = speaker_info[i]
        start_j, duration_j, _ = speaker_info[j]

        end_i = start_i + duration_i
        end_j = start_j + duration_j

        overlap_start = max(start_i, start_j)
        overlap_end = min(end_i, end_j)

        if overlap_start < overlap_end:
            overlap_tuple.append((overlap_start, overlap_end))
    last_end = 0.0
    overlaps = []
    for start, end in overlap_tuple:
        if start <= last_end:
            overlaps.append(end - last_end)
        else:
            overlaps.append(end - start)
        last_end = end
    total_overlap_duration = sum(overlaps)
    relative_overlap = total_overlap_duration / last_end if overlaps != [] else 0.0
    return total_overlap_duration, relative_overlap


def overlap_duration_all_files(rttm_path: str, result_file: str):
    """
    Calculate the total duration of overlapping speech in milliseconds and the relative amount for all files in the
    given directory and writes the result to a csv file.
    :param rttm_path: Path to the rttm directory.
    :param result_file: File to write the result to.
    """
    with open(result_file, 'w') as fp:
        fp.write(f"file, total_overlap_time_ms, relative_overlap_time\n")

    files = [f for f in os.listdir(rttm_path) if os.path.isfile(os.path.join(rttm_path, f)) and
             f.endswith('.rttm')]
    for file in files:
        rttm_file = os.path.join(rttm_path, file)
        total_overlap, relative_overlap = calculate_overlap_duration(rttm_file)
        with open(result_file, 'a') as fp:
            fp.write(f"{file}, {total_overlap}, {relative_overlap}\n")


def calculate_filler_word_amount(txt_file_path: str) -> (int, float):
    """
    Calculate the amount of filler words in a text file.
    :param txt_file_path: Path to the text file.
    :return: The amount of filler words and the relative amount.
    """
    with open(txt_file_path, 'r') as file:
        lines = file.read()

    nlp = spacy.load("en_core_web_trf")
    doc = nlp(lines)
    intj_amount = len([token for token in doc if token.pos_ == "INTJ"])

    number_of_words = len(lines.split())

    return intj_amount, intj_amount / number_of_words


def filler_words_all_files(txt_path: str, result_file: str):
    """
    Calculate the amount of filler words in all txt files in the given directory and writes the result to a csv file.
    :param txt_path: Path to the directory.
    :param result_file: File to write the result to.
    """
    with open(result_file, 'w') as fp:
        fp.write(f"file, total_amount_filler_words, relative_amount_filler_words\n")

    files = [f for f in os.listdir(txt_path) if os.path.isfile(os.path.join(txt_path, f)) and
             f.endswith('.txt')]
    for file in files:
        txt_file = os.path.join(txt_path, file)
        total_filler, relative_filler = calculate_filler_word_amount(txt_file)
        with open(result_file, 'a') as fp:
            fp.write(f"{file}, {total_filler}, {relative_filler}\n")


def mannwhitney(data_a: List[float], data_b: List[float], alternative: str = 'two-sided') -> (int, float):
    """
    Calculates the Mann-Whitney U test for two independent samples.

    :param data_a: Datapoints of a.
    :param data_b: Datapoints of b.
    :param alternative: Defines the alternative hypothesis. Default is 'two-sided', can also be 'greater' or 'less'.
    :return: The Mann-Whitney U statistic and the p-value.
    """
    stat, p = mannwhitneyu(data_a, data_b, alternative=alternative)
    return stat, p


def print_mannwhitney(file_a: str, file_b: str, alternative: str = 'two-sided'):
    """
    Prints the Mann-Whitney U test for two independent samples.

    :param data_a: File path to the first data. Assumes one floating datapoint per line.
    :param data_b: File path to the second data. Assumes one floating datapoint per line.
    :param alternative: Defines the alternative hypothesis. Default is 'two-sided', can also be 'greater' or 'less'.
    """
    data_a = []
    with open(file_a, 'r') as file:
        for line in file:
            data_a.append(float(line.strip()))
    data_b = []
    with open(file_b, 'r') as file:
        for line in file:
            data_b.append(float(line.strip()))
    stat, p = mannwhitney(data_a, data_b, alternative=alternative)
    print(f"Mann-Whitney U statistic: {stat}, p-value: {p}")
