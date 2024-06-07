import getopt
import sys

from code.diarization import diarize_all_files
from code.evaluation import (evaluate_all_files_diarization, overlap_duration_all_files, print_mannwhitney,
                             filler_words_all_files)
from code.forced_alignment import forced_alignment, convert_csv_to_rttm

if __name__ == '__main__':
    arg_count = len(sys.argv)
    if arg_count < 2:
        print('No arguments given. Exiting.')
        sys.exit(1)
    routine = sys.argv[1]
    args = sys.argv[2:]

    opts, _ = getopt.gnu_getopt(args, 'a:r:t:f:w:e:x:y:c', ['audio_path=', 'result_path=',
                                                            'access_token=', 'audio_format=', 'reference_path=',
                                                            'consider_speaker_no', 'dataset_x=', 'dataset_y=',
                                                            'evaluate_file='])

    audio_paths = list(map(lambda x: x[1], filter(lambda x: x[0] == '--audio_path' or x[0] == '-a', opts)))
    audio_path = audio_paths[0] if audio_paths else None
    result_paths = list(map(lambda x: x[1], filter(lambda x: x[0] == '--result_path' or x[0] == '-r', opts)))
    result_path = result_paths[0] if result_paths else None
    access_tokens = list(map(lambda x: x[1], filter(lambda x: x[0] == '--access_token' or x[0] == '-t', opts)))
    access_token = access_tokens[0] if access_tokens else None
    audio_formats = list(map(lambda x: x[1], filter(lambda x: x[0] == '--audio_format' or x[0] == '-f', opts)))
    audio_format = audio_formats[0] if audio_formats else '.wav'
    reference_paths = list(map(lambda x: x[1], filter(lambda x: x[0] == '--reference_path' or x[0] == '-w', opts)))
    reference_path = reference_paths[0] if reference_paths else None
    evaluate_files = list(map(lambda x: x[1], filter(lambda x: x[0] == '--evaluate_file' or x[0] == '-e', opts)))
    evaluate_file = evaluate_files[0] if evaluate_files else None
    dataset_xs = list(map(lambda x: x[1], filter(lambda x: x[0] == '--dataset_x' or x[0] == '-x', opts)))
    dataset_x = dataset_xs[0] if dataset_xs else None
    dataset_ys = list(map(lambda x: x[1], filter(lambda x: x[0] == '--dataset_y' or x[0] == '-y', opts)))
    dataset_y = dataset_ys[0] if dataset_ys else None
    consider_speaker_nos = list(map(lambda x: x[0], filter(lambda x: x[0] == '--consider_speaker_no' or x[0] == '-c',
                                                           opts)))
    consider_speaker_no = True if consider_speaker_nos else False

    if routine == '--diarization' or routine == '-d':
        if not audio_path or not result_path or not access_token:
            print('Missing arguments. Exiting.')
            sys.exit(1)
        if consider_speaker_no:
            if not reference_path:
                print('Reference path needed if number of speakers should be considered. Exiting.')
                sys.exit(1)
            diarize_all_files(audio_path, result_path, access_token, audio_format, True, reference_path)
        else:
            diarize_all_files(audio_path, result_path, access_token, audio_format)
    elif routine == '--forced_alignment' or routine == '-fa':
        if not audio_path or not result_path or not reference_path:
            print('Missing arguments. Exiting.')
            sys.exit(1)
        forced_alignment(audio_path, result_path, reference_path, audio_format)
    elif routine == '--evaluate' or routine == '-e':
        if not audio_path or not result_path or not reference_path or not evaluate_file:
            print('Missing arguments. Exiting.')
            sys.exit(1)
        evaluate_all_files_diarization(reference_path, result_path, audio_path, evaluate_file, audio_format)
    elif routine == '--overlap_calc' or routine == '-o':
        if not reference_path or not evaluate_file:
            print('Missing arguments. Exiting.')
            sys.exit(1)
        overlap_duration_all_files(reference_path, evaluate_file)
    elif routine == '--filler_word_calc' or routine == '-fw':
        if not reference_path or not evaluate_file:
            print('Missing arguments. Exiting.')
            sys.exit(1)
        filler_words_all_files(reference_path, evaluate_file)
    elif routine == '--mannwhitney' or routine == '-mw':
        if not dataset_x or not dataset_y:
            print('Missing arguments. Exiting.')
            sys.exit(1)
        print_mannwhitney(dataset_x, dataset_y)
    elif routine == '--csv_to_rttm' or routine == '-c2r':
        if not reference_path or not result_path:
            print('Missing arguments. Exiting.')
            sys.exit(1)
        convert_csv_to_rttm(reference_path, result_path)
    else:
        print(f'Unknown routine {routine} selected. Exiting.')
