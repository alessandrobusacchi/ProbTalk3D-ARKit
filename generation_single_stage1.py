import logging
import os
import torch
import pickle
import numpy as np
import warnings
from natsort import os_sorted

import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import framework.launch.prepare  # noqa
from framework.model.utils.tools import load_checkpoint, detach_to_numpy
from framework.data.utils import get_split_keyids

logger = logging.getLogger(__name__)

emo_dict = {
       "0": "neutral",   # only have one intensity level
       "1": "happy",
       "2": "sad",
       "3": "surprised",
       "4": "fear",
       "5": "disgusted",
       "6": "angry",
       "7": "contempt"
   }

int_dict = {
       "0": "low",   # only have one intensity level
       "1": "medium",
       "2": "high",
   }


@hydra.main(version_base=None, config_path="configs", config_name="generation_single_firststage")
def _sample(cfg: DictConfig):
    return sample(cfg)


# generate one or multiple samples
def cfg_mean_nsamples_resolution(cfg):
    # If VAE take mean value, set number_of_samples=1
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error("All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    return cfg.number_of_samples == 1


# set VAE variant output path
def get_path_vae(sample_path: Path, onesample: bool, mean: bool, fact: float):
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "none" if fact == 1 else f"{fact}"
    path = sample_path / f"{fact_str}{extra_str}"
    return path


# set VQVAE variant output path
def get_path_vqvae(sample_path: Path, onesample: bool, temperature: float, k: float):
    extra_str = "" if onesample else "_multi"
    tem_str = f"{temperature}"
    k_str = "" if k == 1 else f"_{k}"
    path = sample_path / f"{tem_str}{k_str}{extra_str}"
    return path


# prediction
def sample(newcfg: DictConfig) -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load previous configs
    prevcfg = OmegaConf.load(Path(newcfg.folder) / ".hydra/config.yaml")
    # Merge configs to overload them
    cfg = OmegaConf.merge(prevcfg, newcfg)

    onesample = cfg_mean_nsamples_resolution(cfg)

    logger.info("Sample script. The outputs will be stored in:")
    folder_name = cfg.folder.split("/")[-1]
    output_dir = Path(cfg.path.code_dir) / f"results/generation/{cfg.experiment}/{folder_name}"

    path = None
    if hasattr(cfg.model, 'vqvae_prior') and cfg.model.vqvae_prior:
        path = get_path_vqvae(output_dir, onesample, "none", cfg.k)
    else:
        # FALLBACK FOR STAGE 1 VQ-VAE
        # Since Stage 1 doesn't have prior/pred flags, we just assign a folder
        path = output_dir / "reconstructions"

    # Create the directory safely
    path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Outputs will be saved to: {path}")

    # update the motion prior if needed
    if cfg.folder_prior is not None and cfg.version_prior is not None:
        if os.path.exists(cfg.folder_prior):
            OmegaConf.update(cfg, "model.folder_prior", cfg.folder_prior)
            OmegaConf.update(cfg, "model.version_prior", cfg.version_prior)
        else:
            logger.info(f"Using default motion prior.")

    # save config to check
    OmegaConf.save(cfg, output_dir / "merged_config.yaml")

    from hydra.utils import instantiate
    logger.info("Loading model")
    last_ckpt_path = cfg.last_ckpt_path
    model = instantiate(cfg.model,
                        nfeats=cfg.nfeats,
                        split_path=cfg.data.split_path,
                        one_hot_dim=cfg.data.one_hot_dim,
                        resumed_training=False,
                        logger_name="none",
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    # move model to cuda
    if cfg.device is None:
        device_index = cfg.trainer.devices[0]
    else:
        device_index = cfg.device
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if device_index < num_devices:
            model.to(f"cuda:{device_index}")
        else:
            model.to(f"cuda:0")
    print("device checking:", model.device)

    load_checkpoint(model, last_ckpt_path, eval_mode=True, device=model.device)
    if hasattr(cfg.model, 'vae_pred') and cfg.model.vae_pred:
        model.motion_prior.sample_mean = cfg.mean
        model.motion_prior.fact = cfg.fact
    if hasattr(cfg.model, 'vqvae_pred') and cfg.model.vqvae_pred:
        model.temperature = cfg.temperature
        model.k = cfg.k

    from rich.progress import Progress
    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)

    # set style one hot
    subject = cfg.id
    emotion = cfg.emotion
    intensity = cfg.intensity

    sentence = "001"
    sequence_name = f"{subject}_{sentence}_{emotion}_{intensity}"

    keyid = '{}_x_{}_{}'.format(subject, emotion, intensity)

    gt_motion_path = f"datasets/mead_arkit/param/{sequence_name}.npy"

    if not os.path.exists(gt_motion_path):
        raise FileNotFoundError(f"Could not find ground truth motion at: {gt_motion_path}")

    gt_motion_data = np.load(gt_motion_path)

    motion_tensor = torch.from_numpy(gt_motion_data).unsqueeze(0).float().to(model.device)

    batch = {
        "motion": [motion_tensor],
        "keyid": [keyid]
    }

    with torch.no_grad():
        with Progress(transient=True) as progress:
            task = progress.add_task("Reconstructing motion sequence")

            # Call the model. NO sample=cfg.sample argument!
            prediction = model(batch)

            # The VQVAE forward method returns motion_pred directly
            prediction = prediction.squeeze().detach().cpu().numpy()
            print("Reconstructed shape:", prediction.shape)

            emotion_label = emo_dict[str(emotion)]
            intensity_label = int_dict[str(intensity)]

            # Save the reconstructed sequence
            npypath = path / f"recon_{subject}_{emotion_label}_{intensity_label}.npy"
            np.save(npypath, prediction)

            progress.update(task, advance=1)

    if npypath is not None:
        logger.info(f"Reconstruction complete. You can find it here:\n{npypath.parent}")


if __name__ == '__main__':
    _sample()
