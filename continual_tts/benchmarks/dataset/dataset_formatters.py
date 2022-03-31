import os


def vctk(root_path, meta_file, **kwargs):
    meta_data_path = os.path.join(root_path, meta_file)
    with open(meta_data_path) as file:
        all_lines = file.readlines()
    all_lines = [l.strip() for l in all_lines]
    # Only keep valid speakers
    all_lines = [l.split("|") for l in all_lines]

    def get_track_info(l):
        d = {
            "text": l[2],
            "audio_file": os.path.join(root_path, "wavs", l[0], l[1] + ".wav"),
            "speaker_name": "vctk_" + l[0],
            "language": ""
        }
        return d
    all_lines = [get_track_info(l) for l in all_lines]

    return all_lines


def ljspeech(root_path, meta_file, **kwargs):
    meta_data_path = os.path.join(root_path, meta_file)
    with open(meta_data_path) as file:
        all_lines = file.readlines()
    all_lines = [l.strip() for l in all_lines]
    # Only keep valid speakers
    all_lines = [l.split("|") for l in all_lines]

    def get_track_info(l):
        d = {
            "text": l[2],
            "audio_file": os.path.join(root_path, "wavs", l[0], l[1] + ".wav"),
            "speaker_name": l[0],
            "language": ""
        }
        return d
    all_lines = [get_track_info(l) for l in all_lines]

    return all_lines
