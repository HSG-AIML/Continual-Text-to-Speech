from typing import Dict, List

from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import TTSDataset


class ContinualTTSDataset(TTSDataset):
    def __init__(
            self,
            outputs_per_step: int = 1,
            compute_linear_spec: bool = False,
            ap: AudioProcessor = None,
            samples: List[Dict] = None,
            tokenizer: "TTSTokenizer" = None,
            compute_f0: bool = False,
            f0_cache_path: str = None,
            return_wav: bool = False,
            batch_group_size: int = 0,
            min_text_len: int = 0,
            max_text_len: int = float("inf"),
            min_audio_len: int = 0,
            max_audio_len: int = float("inf"),
            phoneme_cache_path: str = None,
            precompute_num_workers: int = 0,
            speaker_id_mapping: Dict = None,
            d_vector_mapping: Dict = None,
            language_id_mapping: Dict = None,
            use_noise_augment: bool = False,
            start_by_longest: bool = False,
            verbose: bool = False,
    ):
        super(ContinualTTSDataset, self).__init__(
            outputs_per_step=outputs_per_step,
            compute_linear_spec=compute_linear_spec,
            ap=ap,
            samples=samples,
            tokenizer=tokenizer,
            compute_f0=compute_f0,
            f0_cache_path=f0_cache_path,
            return_wav=return_wav,
            batch_group_size=batch_group_size,
            min_text_len=min_text_len,
            max_text_len=max_text_len,
            min_audio_len=min_audio_len,
            max_audio_len=max_audio_len,
            phoneme_cache_path=phoneme_cache_path,
            precompute_num_workers=precompute_num_workers,
            speaker_id_mapping=speaker_id_mapping,
            d_vector_mapping=d_vector_mapping,
            language_id_mapping=language_id_mapping,
            use_noise_augment=use_noise_augment,
            start_by_longest=start_by_longest,
            verbose=verbose,
        )

    def __getitem__(self, idx):
        return self.load_data(idx), -1

    def collate_fn(self, batch):
        batch = [x[0] for x in batch]
        return super().collate_fn(batch)


def get_ctts_dataset(config, samples, is_eval, ap, tokenizer):
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
