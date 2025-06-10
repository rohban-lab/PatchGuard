import os
from urllib.parse import urlparse

def convert_drive_link_to_download(link: str) -> str:
    parsed = urlparse(link)
    if "drive.google.com" not in parsed.netloc:
        raise ValueError("Invalid Google Drive link")
    if "/file/d/" in parsed.path:
        file_id = parsed.path.split("/file/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    raise ValueError("Could not extract file ID")

WEIGHT_LINKS = {
    "mvtec": {
        "bottle": "https://drive.google.com/file/d/1vAV5zzVxx6SqLlirkP0kibEileHxKflz/view?usp=drive_link",
        "cable": "https://drive.google.com/file/d/1mNWWZXmr07bsxOkSY2FerNPM5nSMFIz6/view?usp=drive_link",
        "capsule": "https://drive.google.com/file/d/1THd1QZhCOSbknYMP_lLsALfsn66tt8zh/view?usp=drive_link",
        "carpet": "https://drive.google.com/file/d/1x1N0K33_gaymE6UHmWRYdVem_n2UUJsY/view?usp=drive_link",
        "grid": "https://drive.google.com/file/d/1LvsbOqD2e_GLuOxEtGvlUiq9DGhmGTOX/view?usp=drive_link",
        "hazelnut": "https://drive.google.com/file/d/1F8lFOX19_fhEUJjPnlg_vYkzFwAFa8lt/view?usp=drive_link",
        "leather": "https://drive.google.com/file/d/13_rrkYu4-87zv1TeEngkwTIonBbgTPmO/view?usp=drive_link",
        "metal_nut": "https://drive.google.com/file/d/1MYM56tOgznEGwqUvKxzRedL6gVL5xw8q/view?usp=drive_link",
        "pill": "https://drive.google.com/file/d/1teEY-5VUMOhpTlPzyYaBsW82u0x2-ipg/view?usp=drive_link",
        "screw": "https://drive.google.com/file/d/1Lb8IIRGpBqG7j53Pw0KvPRuAF_Q0YKD8/view?usp=drive_link",
        "tile": "https://drive.google.com/file/d/1vKh-eidKw3bb9l6l9jCA8sqpl0RYqaJ0/view?usp=drive_link",
        "toothbrush": "https://drive.google.com/file/d/1Mcmi4FxqigGv3p-M7lLYX_gDWr1drvvt/view?usp=drive_link",
        "transistor": "https://drive.google.com/file/d/1ap1HRkciis4IWoczqPo_Pr7UZsOMUxwe/view?usp=drive_link",
        "wood": "https://drive.google.com/file/d/1uMzkccZaV4f0yMf0AVt3usiniaNKy5fY/view?usp=drive_link",
    },
    "visa": {
        "candle": "https://drive.google.com/file/d/16LM-735VfnK7Zxa1VJ4xJ3AnpuU11T5F/view?usp=drive_link",
        "capsules": "https://drive.google.com/file/d/1L7EiRCgvcPeAoPEVoxFP0Urk9-z-0fC1/view?usp=drive_link",
        "cashew": "https://drive.google.com/file/d/1hwnmbVyZqPeHj4zD92EB-BKfn9CDs8vU/view?usp=sharing",
        "chewinggum": "https://drive.google.com/file/d/1Lrl26QthpI6dUlIEDAYNnVSju1fyjEdQ/view?usp=drive_link",
        "fryum": "https://drive.google.com/file/d/1YTxzmXCe-sh8dWa1h3bhC4eiPpihG7nW/view?usp=drive_link",
        "macaroni1": "https://drive.google.com/file/d/1UNVTVKmo6ch3D5eeseSwVKUpC-E0mkuY/view?usp=drive_link",
        "macaroni2": "https://drive.google.com/file/d/1tRoyS0dnOUKvGYnUbu2E44269b1CFmx5/view?usp=drive_link",
        "pcb1": "https://drive.google.com/file/d/1nuBmi4J2mNHvYoVO1AFuzTt-nymDEKCq/view?usp=drive_link",
        "pcb2": "https://drive.google.com/file/d/1j5_YR9_qcQP8cwYy3foxgiUUBEiil_nK/view?usp=drive_link",
        "pcb3": "https://drive.google.com/file/d/1p5MAuXvUdg-wqKFSo3lWc6vqU4ochYiW/view?usp=drive_link",
        "pcb4": "https://drive.google.com/file/d/1WPWVcsYq0wEA275_dLN4tB-f_Ri-px_l/view?usp=drive_link",
        "pipe_fryum": "https://drive.google.com/file/d/1JV3Q4lyIz8b98yZyH2aKHlfAg0FM5yla/view?usp=drive_link",
    },
}

def download_weights(dataset_name, class_name, checkpoint_dir):
    dataset_name = dataset_name.lower()
    class_name = class_name.lower()

    if dataset_name not in WEIGHT_LINKS or class_name not in WEIGHT_LINKS[dataset_name]:
        raise ValueError(f"Unknown dataset/class combination: {dataset_name}/{class_name}")

    link = WEIGHT_LINKS[dataset_name][class_name]
    if not link:
        raise ValueError(f"No download link available yet for {dataset_name}/{class_name}")

    download_url = convert_drive_link_to_download(link)

    filename = f"patchguard_{dataset_name}_{class_name}.pth"
    output_path = os.path.join(checkpoint_dir, filename)

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Downloading: {filename} to {output_path}")

    os.system(f'gdown "{download_url}" -O "{output_path}"')

    print("Weight download completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download model weights for a specific dataset/class.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., mvtec, visa)")
    parser.add_argument("--class_name", required=True, help="Class name within the dataset")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory to save the downloaded .pth file")

    args = parser.parse_args()

    download_weights(args.dataset, args.class_name, args.checkpoint_dir)