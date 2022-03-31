from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset
from avalanche.benchmarks import dataset_benchmark

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

from continual_tts.benchmarks.dataset.dataset_formatters import vctk, ljspeech
from continual_tts.benchmarks.dataset.ctts_dataset import get_ctts_dataset


def speaker_incremental_benchmark(
        speaker_lists,
        dataset_config,
        config,
        formatter,
        ap,
        tokenizer
):
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        formatter=formatter,
        eval_split_size=config.eval_split_size
    )

    datasets_train_list = []
    datasets_test_list = []
    for speakers_i in speaker_lists:
        # Train dataset
        train_samples_i = [x for x in train_samples if
                           x["speaker_name"] in speakers_i]

        dataset_train_i = get_ctts_dataset(
            config, train_samples_i, False, ap, tokenizer
        )
        dataset_train_i.targets = [-1] * len(dataset_train_i)
        dataset_train_i.preprocess_samples()
        dataset_train_i_avl = AvalancheDataset(dataset_train_i)
        datasets_train_list.append(dataset_train_i_avl)

        # Test dataset
        test_samples_i = []
        dataset_test_i = get_ctts_dataset(
            config, test_samples_i, True, ap, tokenizer
        )
        dataset_test_i.targets = []
        dataset_eval_i_avl = AvalancheDataset(dataset_test_i)
        datasets_test_list.append(dataset_eval_i_avl)

    benchmark = dataset_benchmark(datasets_train_list, datasets_test_list)

    return benchmark


# ==========> Helper Functions

# VCTK - Speaker Incremental
def get_vctk_spk_inc_benchmark(
        speaker_lists,
        ds_path,
        config,
        ap,
        tokenizer
):
    dataset_config = BaseDatasetConfig(
        name="vctk", path=ds_path, meta_file_train="metadata.txt"
    )

    benchmark = speaker_incremental_benchmark(
        speaker_lists,
        dataset_config,
        config,
        vctk,
        ap,
        tokenizer
    )

    return benchmark


# LJSpeech - Speaker Incremental
def get_ljspeech_spk_inc_benchmark(
        speaker_lists,
        ds_path,
        config,
        ap,
        tokenizer
):
    dataset_config = BaseDatasetConfig(
        name="ljspeech", path=ds_path, meta_file_train="metadata.txt"
    )

    benchmark = speaker_incremental_benchmark(
        speaker_lists,
        dataset_config,
        config,
        ljspeech,
        ap,
        tokenizer
    )

    return benchmark
