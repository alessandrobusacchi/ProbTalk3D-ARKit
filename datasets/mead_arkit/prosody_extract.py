import sys, os
import numpy as np
try:
    import disvoice
except ImportError:
    print("import error for disvoice")
from disvoice.prosody import Prosody

prosody = Prosody()
save_directory_static = "prosody_static/"
save_directory_dynamic = "prosody_dynamic/"
wav_dir = "./wav/"

os.makedirs(save_directory_static, exist_ok=True)
os.makedirs(save_directory_dynamic, exist_ok=True)

audio_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]

# print(audio_files)
count = 0
for file in audio_files:
    file_name = file.split(".")[0]
    target_file_name = file_name + ".npy"
    target_path_static = save_directory_static + target_file_name
    target_path_dynamic = save_directory_dynamic + target_file_name
    audio_file = wav_dir + file
    features_static = prosody.extract_features_file(audio_file, static=True, plots=False, fmt="npy")
    features_dynamic = prosody.extract_features_file(audio_file, static=False, plots=False, fmt="npy")
#    print(target_path)
    np.save(target_path_static, features_static)
    np.save(target_path_dynamic, features_dynamic)
    print("processed: ", audio_files[count])
    count += 1
    print("total processed files: ", count)
    

print("Code ended!")