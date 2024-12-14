# SEARLE Dataset Setup and Command Execution Instructions

This guide provides step-by-step instructions for downloading the necessary datasets, setting up the environment, and running the commands for SEARLE.

## Prerequisites

Before proceeding, ensure you have the following installed on your system:

- Python 3.7 or later
- `pip` package manager
- Basic command-line tools like `wget` and `unzip`

## Instructions for Dataset Download

Follow these steps to download and prepare the datasets required for SEARLE:

### Step 1: Clone the SEARLE Repository

Begin by cloning the SEARLE repository:
```bash
!git clone https://github.com/miccunifi/SEARLE
```

Navigate to the `data` directory:
```bash
cd SEARLE/data/
```

### Step 2: Download Fashion-IQ Dataset

Clone the Fashion-IQ repository:
```bash
!git clone https://github.com/XiaoxiaoGuo/fashion-iq.git
```

### Step 3: Prepare Image Data

Use the following script to download images for the Fashion-IQ dataset. Ensure the `image_url` metadata is available in the appropriate directory.

```python
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

categories = ["asin2url.dress.txt", "asin2url.shirt.txt", "asin2url.toptee.txt"]
base_url_path = "fashion-iq-metadata/image_url"
output_dir = "FashionIQ/images"

os.makedirs(output_dir, exist_ok=True)

def normalize_filename(filename):
    return filename.split()[0]

def download_image(entry, output_dir):
    try:
        file_id, url = entry.strip().split("\t")
        filename = normalize_filename(file_id)
        filepath = os.path.join(output_dir, filename + ".jpg")

        if os.path.exists(filepath):
            return f"Skipped: {filename}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(filepath, "wb") as img_file:
            img_file.write(response.content)

        return f"Downloaded: {filename}"
    except Exception as e:
        return f"Failed to download {entry.strip()}: {e}"

def process_category(category, output_dir):
    url_file = os.path.join(base_url_path, f"{category}")
    with open(url_file, "r") as f:
        entries = f.readlines()
    return entries

def main():
    all_entries = []
    for category in categories:
        all_entries.extend(process_category(category, output_dir))

    with ThreadPoolExecutor(max_workers=1000) as executor:
        futures = {executor.submit(download_image, entry, output_dir): entry for entry in all_entries}

        with tqdm(total=len(all_entries)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                print(result)
                pbar.update(1)

if __name__ == "__main__":
    main()
```

### Step 4: Download COCO Dataset

Run the following commands to download COCO dataset images:
```bash
!wget http://images.cocodataset.org/zips/unlabeled2017.zip
!wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

!unzip unlabeled2017.zip -d COCO2017_unlabeled/
!unzip image_info_unlabeled2017.zip -d COCO2017_unlabeled/
```

### Step 5: Download ImageNet Data
we need to apply in the imagenet website before getting access to the imagenet data at https://image-net.org/index.php
Download ImageNet test data:
```bash
!wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar
!tar -xvf ILSVRC2012_img_test_v10102019.tar -C ImageNet1k/
```

## Instructions to Run Commands After Dataset Construction

Once the datasets are prepared, execute the following commands to process the data and set up additional requirements:

1. Navigate to the dataset directories:
   ```bash
   cd fashion-iq/images/
   !unzip images.zip
   ```

2. Download and prepare unlabeled COCO 2017 images:
   ```bash
   cd ../..
   !git clone https://github.com/miccunifi/CIRCO.git
   cd CIRCO/
   !mkdir COCO2017_unlabeled
   cd COCO2017_unlabeled/
   !unzip image_info_unlabeled2017.zip
   !unzip unlabeled2017.zip
   ```

3. Download and unzip GPT phrases:
   ```bash
   cd /content/SEARLE/data/
   !mkdir GPT_phrases/GPTNeo27B/
   cd GPT_phrases/GPTNeo27B/
   !wget https://github.com/miccunifi/SEARLE/releases/download/weights/GPTNeo27B.zip
   cd ..
   !unzip GPTNeo27B.zip
   ```

4. Install PyTorch and additional dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. Perform image-concept association:
   ```bash
   !python src/image_concepts_association.py --clip-model-name 'ViT-B/32' --dataset imagenet --dataset-path /content/SEARLE/data/ImageNet1k --split test
   !python src/image_concepts_association.py --clip-model-name 'ViT-B/32' --dataset circo --dataset-path /content/SEARLE/data/CIRCO --split test
   ```

6. Run inversion experiments:
   ```bash
   !python src/oti_inversion.py --exp-name 'oti experiment' --clip-model-name 'ViT-B/32' --dataset imagenet --dataset-path /content/SEARLE/data/ImageNet1k --split test
   !python src/oti_inversion.py --exp-name 'oti experiment1' --clip-model-name 'ViT-B/32' --dataset circo --dataset-path /content/SEARLE/data/CIRCO --split test
   ```

7. Validate using SEARLE scripts:
   ```bash
   !python SEARLE/src/validate.py --eval-type searle --dataset circo --dataset-path /content/SEARLE/data/CIRCO/
   ```

## Additional Notes

Ensure the datasets and files are stored in their respective directories as required by SEARLE. For more details, refer to the [official SEARLE repository](https://github.com/miccunifi/SEARLE).

