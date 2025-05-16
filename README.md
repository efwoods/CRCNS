# CRCNS

This is a repository to explore and extrapolate CRCNS data to allow individuals to see and move accordingly to the data identified in this research

# Datasets:

- [Visual Cortex](https://crcns.org/data-sets/vc/pvc-1):
  - Potential Usecases: For curing blindness, visualizing dreams & memories, augmenting vision, and recording/streaming using the human eye. When coupled with the amygdala, this would allow a person to see another's face and feel the experience of love upon gazing at their countenance. This would be a useful application for people with alzheimer's dementia or those in couple's therapy to experience the bonding connection they once had when looking into each other's eyes.
- [Auditory Cortex](https://crcns.org/data-sets/ac):
  - Potential Usecases: For curing deafness, enabling high-quality audio signals at live events, augmenting hearing to enable hearing frequencies outside the normal range of human hearing. Enabling instructions for clandestine operations or domestic phone calls.
- [Motor Cortex](https://crcns.org/data-sets/motor-cortex):
  - Potential Usecases: Predicting movement future movements to understand a Person's intent. Could be used to pre-empt or detect undesired activity such as spasms or violent crime. Could also enable super-human prolonged periods of strength and endurance when applied to feats of strength competitions and running distances.
- [Somatosensory Cortex](https://crcns.org/data-sets/ssc):
  - Potential Usecases: Useful for developing the sensation of touch to allow a person to feel. If a peripheral device was developed, signals from the hands could be sent to the somatosensory cortex to enable people to experience holding hands or cooking or running their hands through a person's hair or performing useful work. This would greatly increase the quality of human life and increase the workforce.

---

Minimum Viable Features for Proof-of-Concept

1. Dataset Loader

Handles secure download, decompression, parsing, and preliminary visualization.

2. Preprocessing Module

Cleans, aligns, and extracts features from neural recordings (LFP, spikes, EEG, etc.)

3. Modeling Module

Trainable modules for each cortex domain with attention to interpretability (e.g. SHAP-compatible models).

4. Simulation Module

Reconstruction of the input (e.g. visual image, audio wave) from neural signals.
✅ Proof-of-Concept: Starter Codebase
📁 crcns_bci

crcns_bci/
├── data_loader.py # Download and parse datasets from CRCNS
├── preprocess.py # Neural signal preprocessing routines
├── train_model.py # Cortex-specific model training
├── simulate.py # Reconstruct / simulate perception
├── api.py # FastAPI backend interface
├── main.py # CLI interface for keyboard-only interaction
├── config.yaml # Dataset paths, model configs
├── requirements.txt # Dependencies
└── README.md

📦 Step 1: data_loader.py

import os
import requests
import zipfile
import tarfile

CRCNS_BASE_URL = "https://crcns.org/files/data"

def download_dataset(dataset_id: str, out_dir: str):
url = f"{CRCNS_BASE_URL}/{dataset_id}/download"
filename = os.path.join(out_dir, f"{dataset_id}.tar.gz")

    print(f"Downloading {dataset_id} from {url}")
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Download complete. Extracting...")
    if filename.endswith("tar.gz") or filename.endswith(".tgz"):
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=out_dir)
    elif filename.endswith(".zip"):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(out_dir)

    print(f"Extraction complete for {dataset_id}")
    return os.path.join(out_dir, dataset_id)

🧪 Step 2: main.py (keyboard-driven CLI interface)

import argparse
from data_loader import download_dataset

def main():
parser = argparse.ArgumentParser(description="CRCNS Neural Cortex Exploration Tool")
parser.add_argument("--dataset", type=str, required=True, help="CRCNS Dataset ID (e.g. pvc-4)")
parser.add_argument("--out", type=str, default="./data", help="Output directory")
args = parser.parse_args()

    dataset_path = download_dataset(args.dataset, args.out)
    print(f"Dataset ready at: {dataset_path}")

if **name** == "**main**":
main()

Usage (no mouse needed):

python main.py --dataset pvc-4 --out ./data

📈 Next Steps

Once you confirm this approach, I will:

    Add preprocessing for Visual Cortex data from pvc-4.

    Build simulation model to recreate an image from the recorded neural firing.

    Integrate auditory decoding from hc-2 as the second pipeline.

    Include FastAPI backend + WebSocket streaming for simulation.

    Add keyboard shortcuts for control.

📚 Citations & Sources

    CRCNS Dataset Repository – https://crcns.org/

    Olshausen & Field (1996) "Emergence of simple-cell receptive field properties by learning a sparse code for natural images" – Nature

    Kay et al. (2008) "Identifying natural images from human brain activity" – Nature

    Gallant Lab UC Berkeley – fMRI decoding work – https://gallantlab.org/

    MNE-Python EEG/MEG Processing – https://mne.tools/

Would you like me to proceed with pvc-4 as the initial target and develop the neural-image reconstruction pipeline next?
