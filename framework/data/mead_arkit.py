import logging
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from rich.progress import track
from .base import BaseDataModule
from .utils import get_split_keyids
from framework.data.tools.collate import audio_normalize
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class MeadARKitDataModule(BaseDataModule):
    def __init__(self, batch_size: int,
                 num_workers: int,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers)
        self.save_hyperparameters(logger=False)

        self.split_path = self.hparams.split_path
        self.one_hot_dim = self.hparams.one_hot_dim
        self.Dataset = MEADARKit
        # Get additional info of the dataset
        sample_overrides = {"split": "train", "tiny": True,
                            "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        self.nfeats = self._sample_set.nfeats


class MEADARKit(Dataset):
    def __init__(self, data_name: str,
                 motion_path: str,
                 audio_path: str,
                 split_path: str,
                 prosody_path: str,
                 load_audio: str,
                 split: str,
                 tiny: bool,
                 progress_bar: bool,
                 debug: bool,
                 **kwargs):
        super().__init__()
        self.data_name = data_name
        self.load_audio = load_audio

        ids = get_split_keyids(path=split_path, split=split)
        if progress_bar:
            enumerator = enumerate(track(ids, f"Loading {data_name} {split} dataset"))
        else:
            enumerator = enumerate(ids)

        if tiny:
            max_data = 2
        elif not tiny and debug:
            max_data = 8
        else:
            max_data = np.inf

        motion_data_all = {}
        shape_data_all = {}
        audio_data_all = {}
        prosody_data_all = {}

        if load_audio:
            for i, id in enumerator:
                if len(motion_data_all) >= max_data:
                    break
                # load 3DMEAD dataset
                key, motion_data, shape_data, audio_data, prosody_data = load_data(keyid=id,
                                                                     motion_path=Path(motion_path),
                                                                     audio_path=Path(audio_path),
                                                                     prosody_path=Path(prosody_path),
                                                                     max_data=max_data,
                                                                     load_audio=load_audio,
                                                                     split=split)
                motion_data_all.update(dict(zip(key, motion_data)))
                shape_data_all.update(dict(zip(key, shape_data)))
                audio_data_all.update(dict(zip(key, audio_data)))
                prosody_data_all.update(dict(zip(key, prosody_data)))
        else:
            for i, id in enumerator:
                if len(motion_data_all) >= max_data:
                    break
                # load 3DMEAD dataset
                key, motion_data, shape_data = load_data(keyid=id,
                                                         motion_path=Path(motion_path),
                                                         audio_path=Path(audio_path),
                                                         prosody_path=Path(prosody_path),
                                                         max_data=max_data,
                                                         load_audio=load_audio,
                                                         split=None)
                motion_data_all.update(dict(zip(key, motion_data)))
                shape_data_all.update(dict(zip(key, shape_data)))

        self.motion_data = motion_data_all
        self.shape_data = shape_data_all
        if load_audio:
            self.audio_data = audio_data_all
            self.prosody_data = prosody_data_all
        self.keyids = list(motion_data_all.keys())
        self.nfeats = self[0]["motion"].shape[2]  # number of features per frame
        print(f"The number of loaded data pair is: {len(self.motion_data)}")
        print(f"Number of features of a motion frame: {self.nfeats}")

    def load_keyid(self, keyid):
        if self.load_audio:
            element = {"motion": self.motion_data[keyid], "shape": self.shape_data[keyid],
                       "audio": self.audio_data[keyid], "prosody": self.prosody_data[keyid],
                       "keyid": keyid}
        else:
            element = {"motion": self.motion_data[keyid], "shape": self.shape_data[keyid], "keyid": keyid}
        return element

    def __getitem__(self, index):
        keyid = self.keyids[index]
        element = self.load_keyid(keyid)
        return element

    def __len__(self):
        return len(self.keyids)

    def __repr__(self):
        return f"{self.data_name} dataset: ({len(self)}, _, ..)"


def load_data(keyid, motion_path, audio_path, prosody_path, max_data, load_audio, split):
    motion_dir = list(motion_path.glob(f"{keyid}*.npy"))
    motion_key = [directory.stem for directory in motion_dir]
    audio_dir = list(audio_path.glob(f"{keyid}*.wav"))
    audio_key = [directory.stem for directory in audio_dir]
    prosody_dir = list(prosody_path.glob(f"{keyid}*.npy"))
    prosody_key = [directory.stem for directory in prosody_dir]

    keys = []
    motion_data = []
    shape_data = []
    audio_data = []
    prosody_data = []
    if load_audio:
        for key in motion_key:
            if len(keys) >= max_data:
                keys = keys[:max_data]
                motion_data = motion_data[:max_data]
                shape_data = shape_data[:max_data]
                audio_data = audio_data[:max_data]
                prosody_data = prosody_data[:max_data]
                break

            if key in audio_key:
                key_split = key.split("_")
                load_key = None
                if split == "train":
                    if int(key_split[2]) == 0 and int(key_split[1]) in range(33):
                        load_key = key
                    elif int(key_split[2]) != 0 and int(key_split[1]) in range(25):
                        load_key = key
                elif split == "val":
                    if int(key_split[2]) == 0 and int(key_split[1]) in range(33, 37):
                        load_key = key
                    elif int(key_split[2]) != 0 and int(key_split[1]) in range(25, 28):
                        load_key = key
                elif split == "test":
                    if int(key_split[2]) == 0 and int(key_split[1]) in range(37, 41):
                        load_key = key
                    elif int(key_split[2]) != 0 and int(key_split[1]) in range(28, 31):
                        load_key = key

                if load_key is not None:
                    m_index = motion_key.index(load_key)
                    m_dir = motion_dir[m_index]
                    m_npy = np.load(m_dir)  # (frames, num_blendshapes)

                    # Use all ARKit blendshapes directly
                    motion_data.append(torch.from_numpy(m_npy[:, :51]).unsqueeze(0))

                    # Shape data empty, ARKit doesn't have extra
                    shape_data.append(torch.zeros((1, m_npy.shape[0], 0)))

                    # Audio
                    a_index = audio_key.index(key)
                    a_dir = audio_dir[a_index]
                    speech_array, _ = librosa.load(a_dir, sr=16000)
                    speech_array = audio_normalize(speech_array)
                    audio_data.append(speech_array)

                    p_index = prosody_key.index(key)
                    p_dir = prosody_dir[p_index]
                    p_npy = np.load(p_dir)
                    if np.isnan(p_npy).any():
                        # print(p_dir, " contains NAN value")
                        p_npy = np.nan_to_num(p_npy)
                    p_torch = torch.from_numpy(p_npy).unsqueeze(0)
                    p_torch = p_torch.to(torch.float32)
                    # prosody_data.append(p_npy)
                    prosody_data.append(p_torch)

                    keys.append(key)
    else:  # ARKit first stage
        for dir in motion_dir:
            if len(keys) >= max_data:
                keys = keys[:max_data]
                motion_data = motion_data[:max_data]
                shape_data = shape_data[:max_data]
                break

            m_npy = np.load(dir)
            keys.append(dir.stem)

            # Ensure 2D
            if m_npy.ndim == 3:
                m_npy = np.squeeze(m_npy, axis=0) if m_npy.shape[0] == 1 else m_npy.reshape(-1, m_npy.shape[-1])
            elif m_npy.ndim == 1:
                m_npy = m_npy[np.newaxis, :]

            motion_data.append(torch.from_numpy(m_npy[:, :51]).unsqueeze(0))
            shape_data.append(torch.zeros((1, m_npy.shape[0], 0)))

    if load_audio:
        return keys, motion_data, shape_data, audio_data, prosody_data
    return keys, motion_data, shape_data