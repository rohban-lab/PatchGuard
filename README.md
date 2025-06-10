# PatchGuard: Adversarially Robust Anomaly Detection and Localization through Vision Transformers and Pseudo Anomalies

Official PyTorch implementation of
["**PatchGuard: Adversarially Robust Anomaly Detection and Localization through Vision Transformers and Pseudo Anomalies**"](https://openaccess.thecvf.com/content/CVPR2025/html/Nafez_PatchGuard_Adversarially_Robust_Anomaly_Detection_and_Localization_through_Vision_Transformers_CVPR_2025_paper.html)  CVPR 2025.
 
**Authors:** *Mojtaba Nafez, Amirhossein Koochakian, Arad Maleki, Jafar Habibi, Mohammad Hossein Rohban*  
**Links:** [ArXiv](), [Proceddings](https://openaccess.thecvf.com/content/CVPR2025/html/Nafez_PatchGuard_Adversarially_Robust_Anomaly_Detection_and_Localization_through_Vision_Transformers_CVPR_2025_paper.html)


<p align="center">
    <img src=motivation-fig.png width="500"> 
</p>

---
## 🔍 Introduction
> Anomaly Detection (AD) and Anomaly Localization (AL) are crucial in fields that demand high reliability, such as medical imaging and industrial monitoring. However, current AD and AL approaches are often susceptible to adversarial attacks due to limitations in training data, which typically include only normal, unlabeled samples. This study introduces PatchGuard, an adversarially robust AD and AL method that incorporates pseudo anomalies with localization masks within a Vision Transformer (ViT)-based architecture to address these vulnerabilities.We begin by examining the essential properties of pseudo anomalies, and follow it by providing theoretical insights into the attention mechanisms required to enhance the adversarial robustness of AD and AL systems. We then present our approach, which leverages Foreground-Aware Pseudo-Anomalies to overcome the deficiencies of previous anomaly-aware methods. Our method incorporates these crafted pseudo-anomaly samples into a ViT-based framework, with adversarial training guided by a novel loss function designed to improve model robustness, as supported by our theoretical analysis.Experimental results on well-established industrial and medical datasets demonstrate that PatchGuard significantly outperforms previous methods in adversarial settings, achieving performance gains of 53.2% in AD and 68.5% in AL, while also maintaining competitive accuracy in non-adversarial settings.

---
## 🛠️ Setup

### Clone the Project
```bash
git clone https://github.com/rohban-lab/PatchGaurd.git
cd PatchGuard
```

### Download Datasets and Foreground Masks
To prepare the datasets for training and evaluation, simply run the following command:
```bash
python download_data.py --dataset <DATASET_NAME>
```
The mask directory, named foreground_mask, will be placed alongside the training and evaluation image folders. For example:
```swift
datasets/MVTec/toothbrush/train/
                            ├── good/
                            └── foreground_mask/
```

### Download Pretrained Weights
Weights for a specific dataset and class can be downloaded by running:
```bash
python download_weight.py --dataset <DATASET_NAME> --class_name <CLASS_NAME> --checkpoint_dir <SAVE_DIR>

```

*Note: Dataset and weight preparation can be automatically triggered by adding the --use_data_prep and --use_weight_prep flags to the training or evaluation commands.*

---
## 🧪 Training

To train the model from scratch:
```bash
python main.py --mode train --class_name <CLASS_NAME> --dataset <DATASET_NAME> --dataset_dir <DATASET_DIR> --epochs <NUM_EPOCHS>
```
---
## 📈 Evaluation
To evaluate a trained model:
```bash
python main.py --mode test --class_name <CLASS_NAME> --dataset <DATASET_NAME> --dataset_dir <DATASET_DIR> --step_test <NUM_ADV_STEPS> --epsilon_test <ADV_EPSILONS_SEQ> --checkpoint_dir <WEIGHT_DIR>
```
---
## 🖼️ Visualization
To visualize localization results:

```bash
python main.py --mode visualization --class_name <CLASS_NAME> --dataset <DATASET_NAME> --dataset_dir <DATASET_DIR> --epsilon_visualization <ADV_EPSILON> --step_visualization <NUM_ADV_STEPS> --checkpoint_dir <WEIGHT_DIR>
```

---
## 📚 Datasets
| Dataset       | Official Page                  | Foreground Mask             | Model Weights               |
|---------------|-------------------------------|----------------------------|----------------------------|
| MvTec AD  | [Link](https://www.mvtec.com/company/research/datasets/mvtec-ad/)                     | [Link](https://drive.google.com/drive/folders/1VHYcJUDja7o2xbYlh7YKYK_6EleizdoU?usp=drive_link)                  | [Link](https://drive.google.com/drive/folders/1Wn_1cE700ORpRmSfDzOfOpYyhAjxsyHr?usp=drive_link)                  |
| VisA | [Link](https://github.com/amazon-science/spot-diff)                     | [Link](https://drive.google.com/drive/folders/1IdLOXyMpi8dzhyeUV6cLOeW5LK4rIOfC?usp=drive_link)                  | [Link](https://drive.google.com/drive/folders/1_MYqikiJvTKp3z_ZHHVjViYF6Yn0oyVu?usp=drive_link)                  |
| MPDD | [Link](https://github.com/stepanje/MPDD)                     | [Link](https://drive.google.com/drive/folders/10JHKrilH8lBwqnM5HRfkt_LytankNJ_S?usp=drive_link)                  | -               |
| BTAD | [Link](https://github.com/pankajmishra000/VT-ADL)                     | [Link](https://drive.google.com/drive/folders/1LGtlVeFbgcC31cQeJnTWX-j2cTpLdsuf?usp=drive_link)                  | -              |
| WFDD | [Link](https://github.com/cqylunlun/GLASS?tab=readme-ov-file#dataset-release)                     | -                | -             |
|  DTD-Synthetic | [Link](https://openaccess.thecvf.com/content/WACV2023/html/Aota_Zero-Shot_Versus_Many-Shot_Unsupervised_Texture_Anomaly_Detection_WACV_2023_paper.html)                     | -                  | -              |
| BraTS2021 | [Link](http://braintumorsegmentation.org/)                     | [Link](https://drive.google.com/file/d/1viMafEbTR2HvMmWWxsrqd7yE9N3VR7FU/view?usp=drive_link)                  | -              |
| HeadCT | [Link](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage)                     | [Link](https://drive.google.com/file/d/1XdUgdMYrfAFydROTWRdj5S14UQbnu725/view?usp=drive_link)                  | -               |


*Note: Links to datasets, masks, and pretrained model weights will be updated here.*

---
## ⚡ Colab Notebook

 An interactive [Colab notebook](https://colab.research.google.com/drive/1Et4LPWpTfIsc3sS4y4nUlVRa1m_esKW3?usp=sharing) is provided for quick experimentation with PatchGuard.

---
## 📚 Citation
If you find this paper and repository helpful in your research, please cite us:
```bibtex
@InProceedings{Nafez_2025_CVPR,
    author    = {Nafez, Mojtaba and Koochakian, Amirhossein and Maleki, Arad and Habibi, Jafar and Rohban, Mohammad Hossein},
    title     = {PatchGuard: Adversarially Robust Anomaly Detection and Localization through Vision Transformers and Pseudo Anomalies},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {20383-20394}
}
```