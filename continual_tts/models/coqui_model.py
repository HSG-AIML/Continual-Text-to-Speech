import os
import copy

from TTS.utils.manage import ModelManager
from TTS.config import load_config
from TTS.vocoder.models import setup_model as setup_vocoder_model
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.models import setup_model as setup_tts_model

from continual_tts.utils.generic import update_config


def _load_vocoder(model_manager, pretrained_vocoder):
    # Pre-trained vocoder
    checkpoint_path_vocoder, config_path_vocoder, model_item_vocoder = \
        model_manager.download_model(pretrained_vocoder)
    vocoder_config = load_config(config_path_vocoder)
    vocoder_model = setup_vocoder_model(vocoder_config)
    vocoder_model.load_checkpoint(vocoder_config,
                                  checkpoint_path_vocoder,
                                  eval=True)
    return vocoder_model, vocoder_config


def _get_model_config_by_name(model_name="tacotron2"):
    if model_name == "tacotron2":
        tts_config = Tacotron2Config
    else:
        raise NotImplementedError()
    return tts_config


def get_coqui_model(params, ds_path):
    # Initialize model manager
    model_manager = ModelManager(
        output_prefix=params["model_manager_output_prefix"]
    )

    # ==========> Vocoder

    vocoder_model, vocoder_config = _load_vocoder(model_manager,
                                                  params["pretrained_vocoder"])

    # ==========> TTS Model

    # Reference TTS model
    if params["pretrained_tts_model"] != "":
        pretrained_checkpoint_path_tts, pretrained_config_path_tts, _ = \
            model_manager.download_model(
            params["pretrained_tts_model"]
        )

        tts_config = load_config(pretrained_config_path_tts)
        tts_model_pretrained = setup_tts_model(config=tts_config)
        tts_model_pretrained.load_checkpoint(tts_config,
                                             pretrained_checkpoint_path_tts,
                                             eval=False)
        tts_config.characters = None

    else:
        tts_config = _get_model_config_by_name(params["model"])()

    # Model config
    config = copy.copy(tts_config)

    # Tokenizer
    config.text_cleaner = "phoneme_cleaners"
    config.use_phonemes = params["use_phonemes"]
    config.phoneme_language = params["phoneme_language"]
    config.phoneme_cache_path = os.path.join(ds_path, "phonemes")
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Speaker embdding
    config.use_d_vector_file = params["use_d_vector_file"]
    config.d_vector_dim = params["d_vector_dim"]

    # Audio
    config.audio = copy.copy(vocoder_config.audio)
    config.audio.sample_rate = params["sample_rate"]
    config.audio.resample = params["resample"]
    ap = AudioProcessor.init_from_config(config)

    # Speaker manager
    d_vectors_file_path = os.path.join(ds_path, "speaker_embedding_means.json")
    speaker_manager = SpeakerManager(d_vectors_file_path=d_vectors_file_path)

    # Update configs with params
    update_config(config, params)

    # Return model
    if params["model"] == "tacotron2":
        tts_model = Tacotron2(
            config,
            ap,
            tokenizer,
            speaker_manager=speaker_manager
        )
    else:
        raise NotImplementedError()

    # Load weights from pre-trained model

    print("Copying model weights from a pre-trained model ...")
    for (n1, p1), (n2, p2) in zip(tts_model.named_parameters(),
                                  tts_model_pretrained.named_parameters()):
        if p1.shape == p2.shape:
            p1.data.copy_(p2.data)
        else:
            print("Could not copy weights for ", n1)

    return config, tts_model, vocoder_model
