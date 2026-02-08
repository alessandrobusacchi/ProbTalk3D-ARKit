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


@hydra.main(version_base=None, config_path="configs", config_name="generation_single")
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
    if hasattr(cfg.model, 'vae_pred') and cfg.model.vae_pred:
        path = get_path_vae(output_dir, onesample, cfg.mean, cfg.fact)
    if hasattr(cfg.model, 'vqvae_pred') and cfg.model.vqvae_pred:
        if not cfg.sample:
            path = get_path_vqvae(output_dir, onesample, "none", cfg.k)
        else:
            path = get_path_vqvae(output_dir, onesample, cfg.temperature, cfg.k)
    if path is None:
        raise ValueError("No model specified in the config file.")
    else:
        path.mkdir(exist_ok=True, parents=True)
        logger.info(f"{path}")

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

    # load audio
    from framework.data.tools.collate import audio_normalize
    import librosa
    audio_dir = Path(cfg.input_path)

    # set style one hot
    subject = cfg.id
    emotion = cfg.emotion
    intensity = cfg.intensity

    print(subject, emotion, intensity)

    from disvoice.prosody import Prosody
    prosody_obj = Prosody()

    name = audio_dir.stem

    speech_array, _ = librosa.load(audio_dir, sr=16000)
    speech_array = audio_normalize(speech_array)
    audio_data = speech_array

    # prosody features
    prosody_features_static = prosody_obj.extract_features_file(audio_dir, static=True, plots=False, fmt="npy")
    if np.isnan(prosody_features_static).any():
        prosody_features_static = np.nan_to_num(prosody_features_static)

    # Ensure prosody is on the same device as the model
    p_torch = torch.from_numpy(prosody_features_static).unsqueeze(0)
    p_torch = p_torch.to(torch.float32).to(model.device)
    prosody_data = p_torch

    keyid = '{}_x_{}_{}'.format(subject, emotion, intensity)

    with torch.no_grad():
        with Progress(transient=True) as progress:
            # Removed 'for i in range(len(name))' loop which was causing multiple redundant runs
            task = progress.add_task("Generating animation for audio")

            print(audio_data)
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()
            print(audio_tensor)
            batch = {"audio": [audio_tensor],
                     "keyid": [keyid],
                     "prosody": [prosody_data]}

            print("Audio:", name, "Emotion:", emotion, "Intensity", intensity)

            prediction = model(batch, sample=cfg.sample)
            prediction = prediction.squeeze()
            prediction = prediction.detach().cpu().numpy()
            print(prediction.shape)

            emotion_label = emo_dict[str(emotion)]
            intensity_label = int_dict[str(intensity)]

            npypath = path / f"{name}_{subject}_{emotion_label}_{intensity_label}.npy"
            np.save(npypath, prediction)

            progress.update(task, advance=1)

    if npypath is not None:
        logger.info(f"All the sampling are done. You can find them here:\n{npypath.parent}")
    else:
        logger.error("No audio input found.")


if __name__ == '__main__':
    _sample()
