import pytorch_lightning as pl
import logging
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.progress import Progress
import pickle
import torch
import numpy as np

import framework.launch.prepare  # noqa
from framework.model.utils.tools import load_checkpoint, detach_to_numpy
from framework.data.tools.collate import collate_motion_and_audio
from deps.flame.flame_pytorch import FLAME
from generation import cfg_mean_nsamples_resolution, get_path_vae, get_path_vqvae


logger = logging.getLogger(__name__)

emo_dict = {
       "0": "neutral",  # only have one intensity level
       "1": "happy",
       "2": "sad",
       "3": "surprised",
       "4": "fear",
       "5": "disgusted",
       "6": "angry",
       "7": "contempt"
   }

int_dict = {
       "0": "low",
       "1": "medium",
       "2": "high",
   }


upper_mask = [
    0, 1, 2, 3, 4,              # Brows
    5, 6, 7,                    # Cheek
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,   # eyes
    49, 50                      # Nose
]

mouth_mask =[
    22, 23, 24, 25,     # Jaw
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48  # Mouth
]


def lve_compute(vertices_gt, vertices_pred, mouth_map):
    vertices_gt = np.array(vertices_gt)
    vertices_pred = np.array(vertices_pred)
    diff = vertices_gt[:, mouth_map] - vertices_pred[:, mouth_map]
    return np.linalg.norm(diff, axis=1)


def seq_std_compute(motion, upper_map):
    # motion shape: (T, 51)
    L2_dis = np.array([np.square(motion[:, v]) for v in upper_map])
    L2_dis = np.transpose(L2_dis, (1, 0))
    L2_dis = np.sum(L2_dis, axis=1)
    L2_dis = np.std(L2_dis, axis=0)
    std = np.mean(L2_dis)
    return std

@hydra.main(version_base=None, config_path="configs", config_name="evaluation")
def _sample(cfg: DictConfig):
    sample(cfg)


def sample(newcfg: DictConfig):
    # Load previous configs

    print("New config: ", newcfg)
    prevcfg = OmegaConf.load(Path(newcfg.folder) / ".hydra/config.yaml")
    # Merge configs to overload them
    cfg = OmegaConf.merge(prevcfg, newcfg)

    onesample = cfg_mean_nsamples_resolution(cfg)

    logger.info("Sample script. The outputs will be stored in:")
    folder_name = cfg.folder.split("/")[-1]
    output_dir = Path(cfg.path.code_dir) / f"results/evaluation/{cfg.experiment}/{folder_name}"
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

    # save config to check
    OmegaConf.save(cfg, output_dir / "merged_config.yaml")
    pl.seed_everything(cfg.seed)

    logger.info("Loading data module")
    # print("My Print: ")
    # cfg.data['prosody_path'] = '${path.datasets}/mead/prosody_static'
    # print("cfg: ", cfg.data)
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.data_name}' loaded")

    logger.info("Loading model")
    last_ckpt_path = cfg.last_ckpt_path
    logger.info(last_ckpt_path)
    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        split_path=data_module.split_path,
                        one_hot_dim=data_module.one_hot_dim,
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

    # load ckpt
    load_checkpoint(model, last_ckpt_path, eval_mode=True, device=model.device)
    if hasattr(cfg.model, 'vae_pred') and cfg.model.vae_pred:
        model.motion_prior.sample_mean = cfg.mean
        model.motion_prior.fact = cfg.fact
    if hasattr(cfg.model, 'vqvae_pred') and cfg.model.vqvae_pred:
        model.temperature = cfg.temperature
        model.k = cfg.k

    # load test data
    dataset = getattr(data_module, f"{cfg.split}_dataset")

    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)

    seq_count = 0
    frame_count = 0
    vertices_all_gt = []        # mve, lve: gt
    vertices_all_pred = []      # mve, lve: first prediction
    mee_all = []                # mean value
    ce_all = []                 # closest value
    motion_std_difference = []  # fdd: first prediction
    abs_motion_std_difference =[] # absolute fdd
    diversity = 0               # 2 subsets
    with torch.no_grad():
        with Progress(transient=True) as progress:
            task = progress.add_task("Sampling", total=len(dataset.keyids))
            for idx, keyid in enumerate(dataset.keyids):
                progress.update(task, description=f"Sampling {keyid}..")

                # load gt data
                motion_gt = np.load(Path(cfg.data.motion_path) / f"{keyid}.npy")    # (1, T, 403)
                # save gt to check (a small part)
                keyid_split = keyid.split("_")
                emo = emo_dict[str(keyid_split[2])]     # retrieve emotion
                ints = int_dict[str(keyid_split[3])]    # retrieve intensity
                gt_path = path / "param" / f"{keyid}_{emo}_{ints}_gt.npy"

                if (idx + 1) % 100 == 0:
                    logger.info(f"Saving param: {gt_path.stem}")
                    gt_path.parent.mkdir(exist_ok=True, parents=True)
                    np.save(gt_path, motion_gt)

                # sample test data
                test_data = dataset.load_keyid(keyid)
                batch = collate_motion_and_audio([test_data])
                
                ce_lve_set = []     # save 10 lve values
                motion_set = []     # save 10 samples
                
                for i in range(cfg.number_of_samples):
                    motion_pred = model(batch.copy(), sample=cfg.sample, generation=False)
                    
                    if i == 0:
                        if motion_gt.shape[0] > motion_pred.shape[1]:
                            motion_gt = motion_gt[:motion_pred.shape[1], :]

                        if motion_pred.shape[1] > motion_gt.shape[0]:
                            motion_pred = motion_pred[:, :motion_gt.shape[0], :]
                    assert motion_gt.shape[0] == motion_pred.shape[1], "Length mismatch"

                    pred_seq = detach_to_numpy(motion_pred.squeeze(0))
                    #print(pred_seq)

                    if cfg.number_of_samples > 1:
                        pred_path = path/"param"/f"{keyid}_{emo}_{ints}_{i}.npy"
                    else:
                        pred_path = path/"param"/f"{keyid}_{emo}_{ints}_one.npy"
                        
                    
                    if (idx+1) % 100 == 0:  
                        if (i+1) % 3 == 1:  
                            logger.info(f"Saving pred: {pred_path.stem}")
                            pred_path.parent.mkdir(exist_ok=True, parents=True)
                            np.save(pred_path, pred_seq)


                    """CE: compute lve for each samples"""
                    ce_lve = lve_compute(vertices_gt=list(motion_gt),
                                         vertices_pred=list(pred_seq),
                                         mouth_map=mouth_mask)
                    ce_lve_set.append(ce_lve)                   # (T,)
                    
                    # save 10 samples
                    motion_set.append(pred_seq)        # (T, 5023, 3)
                    torch.cuda.empty_cache()

                """MVE, LVE: save prediction of all audio samples"""
                vertices_all_gt.extend(list(motion_gt))   # length T of items (5023, 3)
                vertices_all_pred.extend(list(motion_set[0]))   # use the first sample

                """MEE: mean over 10 samples"""
                motion_set_stack = np.stack(motion_set, axis=0)
                vertices_npy_pred_mean = np.mean(motion_set_stack, axis=0)      # (T, 5023, 3)
                mee_lve = lve_compute(vertices_gt=list(motion_gt), 
                                      vertices_pred=list(vertices_npy_pred_mean), 
                                      mouth_map=mouth_mask)
                mee_all.extend(list(mee_lve))

                """CE: closest lve in 10 samples"""
                smallest_lve = None
                smallest_lve_value = float('inf')               # start with an infinitely large value
                for lve_of_one_seq in ce_lve_set:
                    lve_value = np.sum(lve_of_one_seq)
                    if lve_value < smallest_lve_value:
                        smallest_lve_value = lve_value
                        smallest_lve = lve_of_one_seq           # (T,)
                assert smallest_lve is not None, "No smallest distance found"
                ce_all.extend(list(smallest_lve))

                # count sequence, frame numbers
                frame_count += motion_gt.shape[1]
                seq_count += 1

                """FDD computation: use the first sample"""
                upper_std_gt = seq_std_compute(motion=motion_gt, upper_map=upper_mask)
                upper_std_pred = seq_std_compute(motion=motion_set[0], upper_map=upper_mask)
                motion_std_difference.append(upper_std_gt - upper_std_pred)
                abs_motion_std_difference.append(np.abs(upper_std_gt - upper_std_pred))

                """Diversity computation"""
                np.random.shuffle(motion_set)           # list of (T, 5023, 3) number=10
                subset1 = motion_set[:5]
                subset2 = motion_set[5:]
                motion_diversity = 0
                for sample1, sample2 in zip(subset1, subset2):
                    motion_diversity += np.linalg.norm(sample1 - sample2, axis=1).mean()
                if len(subset1) == 5 and len(subset2) == 5:
                    motion_diversity /= len(subset1)
                    diversity += motion_diversity
                else:
                    raise ValueError("Subset length mismatch 5")

                print(f"Done sampling: {keyid} ")       # condition on {cdt_id}")
                torch.cuda.empty_cache()
                progress.update(task, advance=1)

    # logging.disable(logging.NOTSET)
    logger.info('Total sequence number: {}'.format(seq_count))
    logger.info('Total frame number: {}'.format(frame_count))
    
    if seq_count == 0:
        logger.info("No sequences were processed. Unable to compute metrics.")
    else:
        """MVE computation"""
        vertices_all_gt = np.array(vertices_all_gt)     # (frame_cunt, 5023, 3)
        vertices_all_pred = np.array(vertices_all_pred)
        vertices_dis = np.linalg.norm(vertices_all_gt - vertices_all_pred, axis=1)
        logger.info('MVE: {:.4e}'.format(np.mean(vertices_dis)))

        """LVE computation"""
        L2_dis_mouth_max = lve_compute(vertices_gt=vertices_all_gt,
                                       vertices_pred=vertices_all_pred,
                                       mouth_map=mouth_mask)
        logger.info('LVE: {:.4e}'.format(np.mean(L2_dis_mouth_max)))

        """MEE computation"""
        logger.info('MEE: {:.4e}'.format(np.mean(mee_all)))

        """CE computation"""
        logger.info('CE: {:.4e}'.format(np.mean(ce_all)))

        """FDD computation"""
        logger.info('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))
        logger.info('ABS FDD: {:.4e}'.format(sum(abs_motion_std_difference) / len(abs_motion_std_difference)))

        """Divertiy computation"""
        logger.info('Diversity: {:.4e}'.format(diversity / seq_count))

        logger.info(f"All the sampling are done. You can find them here:\n{path}")


if __name__ == '__main__':
    _sample()
