import os
import torch

from pyannote.audio import Pipeline

from code.evaluation import count_speaker_from_rttm


def diarize(audio_file_path: str, result_file_path: str, access_token: str, num_speakers: int = None):
    """
    Applies pyannote.audio's (v.3.1) speaker diarization pipeline to the audio file.
    Writes result in .rttm format to result file.
    The expected number of speakers can be given and if given it is considered during the diarization.

    :param audio_file_path: Path to the audio file.
    :param result_file_path: Path to the result file.
    :param access_token: Huggingface access token.
    :param num_speakers: Number of speakers that the diarization should find. Default is None.
    """
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token=access_token)
    if torch.cuda.is_available():
        pipeline.to(torch.device('cuda'))  # switch to gpu

    # Checks sample_rate and adapts
    diarization = pipeline(audio_file_path, num_speakers=num_speakers)
    with open(result_file_path, "w") as fp:
        diarization.write_rttm(fp)


def diarize_all_files(audio_path: str, result_path: str, access_token: str, audio_format: str = '.wav',
                      consider_speakers: bool = False, reference_path: str = None):
    """
    Applies pyannote.audio's (v.3.1) speaker diarization pipeline to all audio files in the given directory.
    Writes result in .rttm format to the result path. If consider_speakers is True, the number of speakers given is
    during the diarization process. In that case, the reference path must be given and the number of speakers is taken
    from the respective rttm files there, that need to have the same name as the audio files.

    :param audio_path: Path to the audio files.
    :param result_path: Path to the result files.
    :param access_token: Huggingface access token.
    :param audio_format: Format of the audio files. Default is '.wav'.
    :param consider_speakers: If True, the number of speakers is considered during diarization. Default is False.
    :param reference_path: Path to the reference rttm files, only needed if consider_speakers is True to determine the
    number of speakers. Default is None.
    """
    wav_files = [f for f in audio_path if os.path.isfile(os.path.join(audio_path, f)) and f.endswith(audio_format)]

    for wav_file in wav_files:
        rttm_file = wav_file.split('.')[0] + '.rttm'
        result_file = os.path.join(result_path, rttm_file)
        audio_file = os.path.join(audio_path, wav_file)
        num_speakers = None
        if consider_speakers:
            num_speakers = count_speaker_from_rttm(os.path.join(reference_path, rttm_file))
        diarize(audio_file, result_file, access_token, num_speakers=num_speakers)