# Continual-Text-to-Speech (TTS)

TBA: Introduction

## Structure
The code structure is inspired by Avalanche. The goal is to extend it by adding
new modules to the corresponding folders.

    
    ├── continual_tts                    # ContinualTTS package
        ├── benchmarks                   # Everything related to data handling
            ├── ...                      # add new benchmarks here
            ├── speaker_incremental.py   # speaker-incremental scenario

        ├── metrics                      # Metrics and Logging
            ├── ...                      # add new metrics here
            ├── loss_metric.py           # loss metric

        ├── models                       # Models
            ├── ...                      # add new models here
            ├── coqui_model.py           # loader for Coqui TTS models    
        
        ├── plugins                      # Plugins (for Avalanche strategies)
            ├── ...                      # add new plugins here
            ├── sample_syntehsize.py     # plugin for synthesizing samples

        ├── strategies
            ├── templates                # base strategy templates   
            ├── ...                      # add new strategies here
            ├── naive.py                 # naive (fine-tuning) strategy

        ├── trainer
            ├── templates                # base trainer templates
            ├── iid_trainer.py           # IID trainer
            ├── spk_inc_trainer.py       # speaker-incremental trainer

        ├── utils
            ├── ...                      # various utils and helpers

    ├── train.py                         # train function: strating point
                                         # which loads params, initializes trainer
                                         # and runs the trainer
    
    ├── hparams                       
        ├── ...                       # all experiment parameters go here
                    

## Plugins

- `gradient_clipper`: clips gradient norms during trainer (before every update).
- `sample_synthesizer`: synthesizes samples after particular number of epochs.