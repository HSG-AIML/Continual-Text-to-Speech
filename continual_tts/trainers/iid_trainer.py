import os
import torch
from tqdm import tqdm
import wandb
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from continual_tts.utils.plot_utils import plot_spectrogram, plot_attention
from .templates import BaseTrainer


class Trainer(BaseTrainer):
    """
    Base class Trainer. All trainers should inherit from this class.
    """
    def __init__(self, args, params, experiment_name):
        super().__init__(args, params, experiment_name)

        self.model.to(self.device)
        self.config.log_to_wandb = args.wandb_proj != ""
        if self.config.log_to_wandb:
            wandb.init(
                project=self.args.wandb_proj,
                config=self.params,
                name=experiment_name)
        self.global_step = 0

    def run(self):
        dataset = self.benchmark.train_stream[0].dataset

        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=self.config.batch_size,
            sampler=None,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )

        for epoch in range(self.config.epochs):
            pbar = tqdm(dataloader)

            epoch_losses = []
            for mbatch in pbar:
                self._unpack_minibatch(mbatch)
                self.optimizer.zero_grad()
                outputs, loss_dict = self.model.train_step(
                    self.mbatch,
                    self.criterion
                )
                loss = loss_dict["loss"]
                loss.backward()

                if self.model.config.grad_clip > 0.0:
                    grad_norm = clip_grad_norm_(self.model.parameters(),
                                                self.model.config.grad_clip)
                self.optimizer.step()

                # Logging
                pbar.set_description(
                    f"Epoch: {epoch} - Loss: {loss.item():.2f}")
                if self.config.log_to_wandb:
                    wandb.log({"Step Loss": loss.item()}, step=self.global_step)
                epoch_losses.append(loss.item())

                self.global_step += 1

            if self.config.log_to_wandb:
                epoch_loss = np.mean(epoch_losses)
                wandb.log({"Epoch Loss": epoch_loss}, step=self.global_step)

            if self.args.save_results:
                if epoch % 5 == 0:
                    self.save_checkpoint(epoch)

            if epoch % self.config.synthesize_samples_every == 0:
                self.synthesize_samples(current_epoch=epoch)

        if self.config.log_to_wandb:
            wandb.finish()

    def _unpack_minibatch(self, mbatch):
        """Move to device"""
        self.mbatch = self.model.format_batch(mbatch)

        # Add speaker embedding to the batch
        speaker_embeddings = [
            self.model.speaker_manager.get_d_vectors_by_speaker(spk) for spk in
            self.mbatch["speaker_names"]]
        speaker_embeddings = torch.FloatTensor(speaker_embeddings).squeeze(1)
        self.mbatch["d_vectors"] = speaker_embeddings.to(self.device)

        # Move to compute device
        for k in self.mbatch.keys():
            if isinstance(self.mbatch[k], torch.Tensor):
                self.mbatch[k] = self.mbatch[k].to(self.device)

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoints_path,
                                       f"ckpt_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self):
        # Load checkpoint
        print(f"Loading checkpoint from  " + \
              f"{self.params['load_checkpoint_path']}")
        ckpt = torch.load(self.params["load_checkpoint_path"],
                          map_location=self.device)
        for name, param in self.model.named_parameters():
            try:
                self.model.state_dict()[name].copy_(ckpt[name])
            except:
                print(f"Could not load weights for {name}")

    def synthesize_samples(self, current_epoch):
        for spk in self.model.config.speaker_lists[0]:
            spk_emb = \
                self.model.speaker_manager.get_d_vectors_by_speaker(spk)
            spk_emb = torch.FloatTensor(spk_emb).squeeze(1).to(self.device)

            text = self.model.config.test_sentences[0]
            text_inputs = self.model.tokenizer.text_to_ids(text)
            text_inputs = \
                torch.LongTensor(text_inputs).unsqueeze(0).to(self.device)

            aux_input = {"d_vectors": spk_emb}

            self.model.eval()
            with torch.no_grad():
                outputs = self.model.inference(text_inputs,
                                                   aux_input=aux_input)
            self.model.train()

            if self.model.config.log_to_wandb:
                mel = outputs["model_outputs"][0].detach().cpu().numpy().T
                wav = self.model.ap.inv_melspectrogram(mel)
                wandb.log(
                    {f"Audio-Epoch{current_epoch}-{spk}": wandb.Audio(
                        wav,
                        caption=text,
                        sample_rate=self.model.ap.sample_rate)},
                    step=self.global_step)

                fig = plot_spectrogram(mel)
                wandb.log(
                    {f"Mel-Epoch{current_epoch}-{spk}": fig},
                    step=self.global_step
                )

                attn = outputs["alignments"][0]
                fig = plot_attention(attn.cpu().numpy())
                wandb.log(
                    {f"Attn-Epoch{current_epoch}-{spk}": fig},
                    step=self.global_step
                )

            # synthesize only for the first speaker
            break
