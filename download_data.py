import os
import tarfile
import zipfile
import shutil

def download_and_extract(url, dest_path, ext_name, archive_format="tar"):
    os.makedirs(dest_path, exist_ok=True)
    filepath = f"{dest_path}/{ext_name}"
    
    if not os.path.exists(filepath):
        print(f"Downloading {ext_name} ...")
        os.system(f'gdown "{url}" -O "{filepath}"')
        print("Download completed.")
    else:
        print(f"The url is already downloaded.")

    print(f"Extracting {ext_name} ...")
    if archive_format == "tar":
        with tarfile.open(filepath, "r:*") as tar:
            tar.extractall(dest_path)
    elif archive_format == "zip":
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(dest_path)
    print("Extraction complete.")

    os.remove(filepath)

def download_dataset(dataset):
    dataset = dataset.lower()

    if dataset == "mvtec":
        final_root = "datasets/MVTec"
        if os.path.exists(final_root):
            print(f"MVTec dataset already exists at {final_root}. Skipping download.")
        else:
            mvtec_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
            download_and_extract(mvtec_url, final_root, f"{dataset}.tar", "tar")

        fg_zip_url = "https://drive.google.com/uc?export=download&id=18Ml7zpIw47wWhlV9OLYl3pWPeQTK1QiI"
        temp_fg_path = os.path.join(final_root, "temp_fg")
        download_and_extract(fg_zip_url, temp_fg_path, "MVTec_FG.zip", archive_format="zip")

        for class_name in os.listdir(temp_fg_path):
            src = os.path.join(temp_fg_path, class_name)
            dst = os.path.join(final_root, class_name, "train", "foreground_mask")
            if os.path.exists(dst):
                print(f"Foreground mask for {class_name} already exists. Skipping.")
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                print(f"Moved mask for {class_name} to {dst}")

        shutil.rmtree(temp_fg_path)

    elif dataset == "visa":
        final_root = "datasets/VisA"
        if os.path.exists(final_root):
            print(f"VisA dataset already exists at {final_root}. Skipping download.")
        else:
            visa_url = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
            download_and_extract(visa_url, final_root, f"{dataset}.tar", "tar")

        fg_zip_url = "https://drive.google.com/uc?export=download&id=1c4oLDSiuhmejGspI17TtgaggLSlNL-Pi"
        temp_fg_path = os.path.join(final_root, "temp_fg")
        download_and_extract(fg_zip_url, temp_fg_path, "VisA_FG.zip", archive_format="zip")

        for class_name in os.listdir(temp_fg_path):
            src = os.path.join(temp_fg_path, class_name)
            dst = os.path.join(final_root, class_name, "Data", "Images", "foreground_mask")
            if os.path.exists(dst):
                print(f"Foreground mask for {class_name} already exists. Skipping.")
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                print(f"Moved mask for {class_name} to {dst}")

        shutil.rmtree(temp_fg_path)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["mvtec", "visa"])
    args = parser.parse_args()

    download_dataset(args.dataset)
