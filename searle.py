# -*- coding: utf-8 -*-
"""SEARLE.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Pfi9n25SpehnSk2q_iIfAsc8VTK1G5g-
"""

!git clone https://github.com/miccunifi/SEARLE

!pip install comet-ml==3.33.6 transformers==4.24.0 tqdm pandas==1.4.2
!pip install git+https://github.com/openai/CLIP.git
!pip install pytorch==1.11.0
!pip install torchvision==0.12.0

cd SEARLE/data/

!git clone https://github.com/XiaoxiaoGuo/fashion-iq.git

import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # For progress display

# Paths
categories = ["asin2url.dress.txt", "asin2url.shirt.txt", "asin2url.toptee.txt"]
base_url_path = "fashion-iq-metadata/image_url"
output_dir = "FashionIQ/images"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def normalize_filename(filename):
    # Remove special characters and keep only the ID (up to the first whitespace or tab)
    return filename.split()[0]

def download_image(entry, output_dir):
    try:
        # Parse the entry: ID and URL
        file_id, url = entry.strip().split("\t")
        filename = normalize_filename(file_id)
        filepath = os.path.join(output_dir, filename + ".jpg")

        # Skip if the file already exists
        if os.path.exists(filepath):
            return f"Skipped: {filename}"

        # Download the image
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        # Save the image
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

    # Create a thread pool with 1000 threads
    with ThreadPoolExecutor(max_workers=1000) as executor:
        futures = {executor.submit(download_image, entry, output_dir): entry for entry in all_entries}

        # Display progress with tqdm
        with tqdm(total=len(all_entries)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                print(result)  # Print the result (Downloaded/Failed/Skipped)
                pbar.update(1)

if __name__ == "__main__":
    main()

# !cp /content/drive/MyDrive/SEARLE/data/fashioniq/images.zip ./fashion-iq/images.zip

!pwd

cd fashion-iq/

cd images/

!unzip images.zip

!wget http://images.cocodataset.org/zips/unlabeled2017.zip

!wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

!mv image_info_unlabeled2017.zip ../

!pwd

cd ../..

!git clone https://github.com/miccunifi/CIRCO.git

cd CIRCO/

!mkdir COCO2017_unlabeled

cd COCO2017_unlabeled/

!unzip image_info_unlabeled2017.zip

!unzip unlabeled2017.zip

!pwd

cd /content/SEARLE/data/

!mkdir GPT_phrases/GPTNeo27B/

cd GPT_phrases/GPTNeo27B/

!wget https://github.com/miccunifi/SEARLE/releases/download/weights/GPTNeo27B.zip

cd ..

!unzip GPTNeo27B.zip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

!cp /content/drive/MyDrive/ILSVRC2012_img_test_v10102019.tar /content/SEARLE/data/

!wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar

!mv -r /content/SEARLE/test

!ls /content/SEARLE/data/ImageNet1k/

!python src/image_concepts_association.py --clip-model-name 'ViT-B/32' --dataset imagenet --dataset-path /content/SEARLE/data/ImageNet1k --split test

!python src/image_concepts_association.py --clip-model-name 'ViT-B/32' --dataset circo --dataset-path /content/SEARLE/data/CIRCO --split test

!python src/oti_inversion.py --exp-name 'oti experiment' --clip-model-name 'ViT-B/32' --dataset imagenet --dataset-path /content/SEARLE/data/ImageNet1k --split test

!python src/oti_inversion.py --exp-name 'oti experiment1' --clip-model-name 'ViT-B/32' --dataset circo --dataset-path /content/SEARLE/data/CIRCO --split test

!python SEARLE/src/validate.py --eval-type searle  --dataset circo --dataset-path /content/SEARLE/data/CIRCO/
