from .speaker_incremental import get_vctk_spk_inc_benchmark, \
    get_ljspeech_spk_inc_benchmark


def get_benchmark(params, ds_path, config, ap, tokenizer):

    if params["benchmark"] == "vctk-spk-inc":
        benchmark = get_vctk_spk_inc_benchmark(
            params["speaker_lists"],
            ds_path,
            config,
            ap,
            tokenizer
        )
    elif params["benchmark"] == "ljspeech-spk-inc":
        benchmark = get_ljspeech_spk_inc_benchmark(
            params["speaker_lists"],
            ds_path,
            config,
            ap,
            tokenizer
        )
    else:
        raise NotImplementedError()

    return benchmark
