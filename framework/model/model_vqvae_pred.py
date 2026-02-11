import os
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List
from hydra.utils import instantiate
import logging

from framework.model.utils.tools import load_checkpoint, create_one_hot, resample_input
from framework.model.metrics.compute import ComputeMetrics
from framework.model.base import BaseModel
from framework.data.utils import get_split_keyids

logger = logging.getLogger(__name__)


def adjust_input_representation(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    """
    Brings audio embeddings and visual frames to the same frame rate.

    Args:
        audio_embedding_matrix: The audio embeddings extracted by the audio encoder
        vertex_matrix: The animation sequence represented as a series of vertex positions (or blendshape controls)
        ifps: The input frame rate (it is 50 for the HuBERT encoder)
        ofps: The output frame rate
    """
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]

        if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
            audio_embedding_matrix = audio_embedding_matrix[:, :vertex_matrix.shape[1] * 2]

        elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2:
            vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1] // 2]
    elif ifps > ofps:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True,
                                               mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
    else:
        factor = 1
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True,
                                               mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (
    1, audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))
    return audio_embedding_matrix, vertex_matrix, frame_num


class VqvaePredict(BaseModel):
    def __init__(self,
                 # passed trough datamodule
                 nfeats: int,
                 split_path: str,
                 one_hot_dim: List,
                 resumed_training: bool,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.nfeats = nfeats
        self.resumed_training = resumed_training
        self.working_dir = self.hparams.working_dir

        self.feature_extractor = instantiate(self.hparams.feature_extractor)
        logger.info(f"1. Audio feature extractor '{self.feature_extractor.hparams.name}' loaded")
        audio_encoded_dim = self.feature_extractor.audio_encoded_dim    # 768

        # Style one-hot embedding
        self.all_identity_list = get_split_keyids(path=split_path, split="train")
        self.all_identity_onehot = torch.eye(len(self.all_identity_list))

        # Load motion prior
        self.motion_prior = instantiate(self.hparams.motion_prior,
                                        nfeats=self.hparams.nfeats,
                                        logger_name="none",
                                        resumed_training=False,
                                        _recursive_=False)
        logger.info(f"2. '{self.motion_prior.hparams.modelname}' loaded")
        # Load the motion prior in eval mode
        if os.path.exists(self.hparams.ckpt_path_prior):
            load_checkpoint(model=self.motion_prior,
                            ckpt_path=self.hparams.ckpt_path_prior,
                            eval_mode=True,
                            device=self.device)
        else:
            raise ValueError(f"Motion Autoencoder path not found: {self.hparams.ckpt_path_prior}")
        for param in self.motion_prior.parameters():
            param.requires_grad = False

        self.feature_predictor = instantiate(self.hparams.feature_predictor,
                                             audio_dim=audio_encoded_dim * 2,   # 768*2
                                             one_hot_dim=sum(one_hot_dim[:]),   # 24+8+3
                                             prosody_dim=103)
        logger.info(f"3. 'Audio Encoder' loaded")

        self.optimizer = instantiate(self.hparams.optim, params=self.parameters())
        self._losses = torch.nn.ModuleDict({split: instantiate(self.hparams.losses, _recursive_=False)
                                            for split in ["losses_train", "losses_test", "losses_val"]})

        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        self.metrics = ComputeMetrics()

        # If we want to override it at testing time
        self.temperature = 0.2
        self.k = 1

        self.__post_init__()

    # Forward: audio => motion, called during sampling
    def forward(self, batch, sample, generation=True) -> Tensor:
        self.feature_predictor.to(self.device)
        self.feature_extractor.to(self.device)
        self.motion_prior.to(self.device)

        # style embedding
        style_one_hot = create_one_hot(keyids=batch["keyid"],
                                       IDs_list=self.all_identity_list,
                                       IDs_labels=self.all_identity_onehot,
                                       one_hot_dim=self.hparams.one_hot_dim)

        # audio feature extraction
        audio_features = self.feature_extractor(batch['audio'], False) # list of [B, Ts, 768]

        prosody_feature = []
        resampled_audio_feature = []
        resampled_motion_feature = []

        assert len(audio_features) == 1, "Batch size > 1 not supported"

        for idx in range(len(audio_features)):
            audio_feat = audio_features[idx]  # [1, Ts, 768]
            prosody_feature.append(batch['prosody'][idx])

            if not generation:
                motion_ref = batch['motion'][idx]  # [1, T, 53]
            else:
                # estimate target motion length at 30 fps
                T_audio = audio_feat.shape[1]  # 50 fps
                T_motion = int(T_audio / 50 * 30)
                motion_ref = torch.zeros(
                    (1, T_motion, self.nfeats),
                    device=audio_feat.device
                )

            audio_feat, motion_ref, frame_num = adjust_input_representation(
                audio_feat,
                motion_ref,
                ifps=50,
                ofps=30
            )

            resampled_audio_feature.append(audio_feat[:, :frame_num])
            resampled_motion_feature.append(motion_ref[:, :frame_num])

        # only works for batch_size = 1
        batch['audio'] = torch.cat(resampled_audio_feature, dim=0).float()
        batch['prosody'] = torch.cat(prosody_feature, dim=0).float()

        # predictor
        prediction = self.feature_predictor(
            batch['audio'],
            style_one_hot.to(self.device),
            batch['prosody'].to(self.device)
        )  # [B, T, 256]

        motion_quant_pred, _, _ = self.motion_prior.quantize(
            prediction,
            sample=sample,
            temperature=self.temperature,
            k=self.k
        )

        motion_out = self.motion_prior.motion_decoder(motion_quant_pred)  # [B, T, 53]
        return motion_out


    # Called during training
    def allsplit_step(self, split: str, batch, batch_idx):
        # extract audio features
        audio_features = self.feature_extractor(batch['audio'], False)  # list of [1, Ts, 768]

        # style embedding
        style_one_hot = create_one_hot(
            keyids=batch["keyid"],
            IDs_list=self.all_identity_list,
            IDs_labels=self.all_identity_onehot,
            one_hot_dim=self.hparams.one_hot_dim
        )

        resampled_audio_feature = []
        resampled_motion_feature = []
        prosody_feature = []

        assert len(audio_features) == 1, "Batch size > 1 not supported"

        for idx in range(len(audio_features)):
            audio_feat = audio_features[idx]  # [1, Ts, 768]
            motion_feat = batch['motion'][idx]  # [1, Tm, 53]
            prosody_feature.append(batch['prosody'][idx])

            # 🔑 unified resampling (same as inference)
            audio_feat, motion_feat, frame_num = adjust_input_representation(
                audio_feat,
                motion_feat,
                ifps=50,
                ofps=30
            )

            resampled_audio_feature.append(audio_feat[:, :frame_num])
            resampled_motion_feature.append(motion_feat[:, :frame_num])

        # only works for batch_size = 1
        batch['audio'] = torch.cat(resampled_audio_feature, dim=0).float()  # [1, T, ?]
        batch['motion'] = torch.cat(resampled_motion_feature, dim=0).float()  # [1, T, 53]
        batch['prosody'] = torch.cat(prosody_feature, dim=0).float()

        # predictor forward
        prediction = self.feature_predictor(
            batch['audio'],
            style_one_hot.to(self.device),
            batch['prosody'].to(self.device)
        )  # [B, T, 256]

        # VQ-VAE prior
        motion_quant_pred, _, _ = self.motion_prior.quantize(prediction)
        motion_pred = self.motion_prior.motion_decoder(motion_quant_pred)

        # reference quantization
        motion_quant_ref, _ = self.motion_prior.get_quant(batch['motion'])
        motion_ref = batch['motion']

        assert motion_pred.shape == motion_ref.shape, \
            "Dimension mismatch between prediction and reference motion."

        loss = self.losses[split].update(
            motion_quant_pred=motion_quant_pred,
            motion_quant_ref=motion_quant_ref,
            motion_pred=motion_pred,
            motion_ref=motion_ref
        )

        # metrics
        if split == "val":
            self.metrics.update(
                motion_pred.detach(),
                motion_ref.detach(),
                [motion_ref.shape[1]] * motion_ref.shape[0]
            )

        # logging
        self.allsplit_batch_end(split, batch_idx)

        if "total/train" in self.trainer.callback_metrics:
            self.log(
                "loss_train",
                self.trainer.callback_metrics["total/train"].item(),
                prog_bar=True,
                on_step=True,
                on_epoch=False
            )

        if "total/val" in self.trainer.callback_metrics:
            self.log(
                "loss_val",
                self.trainer.callback_metrics["total/val"].item(),
                prog_bar=True,
                on_step=True,
                on_epoch=False
            )

        return loss