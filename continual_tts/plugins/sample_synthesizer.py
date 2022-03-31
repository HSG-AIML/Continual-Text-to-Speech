import torch
import wandb

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from continual_tts.utils.plot_utils import plot_spectrogram,  plot_attention


class SampleSynthesizer(SupervisedPlugin):
    """
    This plugins synthesize samples after particular number of epochs
    during training.
    """

    def after_training_epoch(
            self, strategy, *args, **kwargs
    ):
        train_exp_epochs = strategy.clock.train_exp_epochs
        synthesize_samples_every = \
            strategy.model.config.synthesize_samples_every
        if train_exp_epochs % synthesize_samples_every == 0:
            self._synthesize_samples(strategy)

    def _synthesize_samples(self, strategy):
        current_exp = strategy.clock.train_exp_counter
        for speakers in strategy.model.config.speaker_lists[:current_exp+1]:
            spk = speakers[0]
            spk_emb = \
                strategy.model.speaker_manager.get_d_vectors_by_speaker(spk)
            spk_emb = torch.FloatTensor(spk_emb).squeeze(1).to(strategy.device)

            text = strategy.model.config.test_sentences[0]
            text_inputs = strategy.model.tokenizer.text_to_ids(text)
            text_inputs = \
                torch.LongTensor(text_inputs).unsqueeze(0).to(strategy.device)

            aux_input = {"d_vectors": spk_emb}

            strategy.model.eval()
            with torch.no_grad():
                outputs = strategy.model.inference(text_inputs,
                                                   aux_input=aux_input)
            strategy.model.train()

            if strategy.model.config.log_to_wandb:
                mel = outputs["model_outputs"][0].detach().cpu().numpy().T
                mel = outputs["model_outputs"][0].detach().cpu().numpy().T
                mel = strategy.model.ap.denormalize(mel).T
                vocoder_input = strategy.model.vocoder.ap.normalize(mel.T)
                vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)
                waveform = strategy.model.vocoder.inference(vocoder_input)
                waveform = waveform.cpu().squeeze().numpy()

                step = strategy.clock.train_iterations
                wandb.log(
                    {f"Audio-Exp{current_exp}-{spk}": wandb.Audio(
                        waveform,
                        caption=text,
                        sample_rate=strategy.model.ap.sample_rate)},
                    step=step)

                fig = plot_spectrogram(mel)
                wandb.log(
                    {f"Mel-Exp{current_exp}-{spk}": fig},
                    step=step
                )

                attn = outputs["alignments"][0]
                fig = plot_attention(attn.cpu().numpy())
                wandb.log(
                    {f"Attn-Exp{current_exp}-{spk}": fig},
                    step=step
                )
