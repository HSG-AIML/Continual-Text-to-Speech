from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset
from avalanche.benchmarks import dataset_benchmark

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

from continual_tts.benchmarks.dataset.dataset_formatters import vctk, ljspeech
from dataset.ctts_dataset import ContinualTTSDataset


def get_tts_dataset(config, samples, is_eval, ap, tokenizer):
    dataset = ContinualTTSDataset(
        outputs_per_step=config.r if "r" in config else 1,
        compute_linear_spec=config.model.lower() == "tacotron" or
                            config.compute_linear_spec,
        compute_f0=config.get("compute_f0", False),
        f0_cache_path=config.get("f0_cache_path", None),
        samples=samples,
        ap=ap,
        return_wav=config.return_wav if "return_wav" in config else False,
        batch_group_size=0 if is_eval else
        config.batch_group_size * config.batch_size,
        min_text_len=config.min_text_len,
        max_text_len=config.max_text_len,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        phoneme_cache_path=config.phoneme_cache_path,
        precompute_num_workers=config.precompute_num_workers,
        use_noise_augment=False if is_eval else config.use_noise_augment,
        verbose=False,
        # speaker_id_mapping=speaker_id_mapping,
        # d_vector_mapping=d_vector_mapping if
        # config.use_d_vector_file else None,
        tokenizer=tokenizer,
        start_by_longest=config.start_by_longest,
        # language_id_mapping=language_id_mapping,
    )

    return dataset


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

        dataset_train_i = get_tts_dataset(
            config, train_samples_i, False, ap, tokenizer
        )
        dataset_train_i.targets = [-1] * len(dataset_train_i)
        dataset_train_i.preprocess_samples()
        dataset_train_i_avl = AvalancheDataset(dataset_train_i)
        datasets_train_list.append(dataset_train_i_avl)

        # Test dataset
        test_samples_i = []
        dataset_test_i = get_tts_dataset(
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
