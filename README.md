## ISIC: Skin Lesion Classification

![ISIC Dataset](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Melanoma.jpg/800px-Melanoma.jpg)  

This repository contains code for skin lesion classification using deep learning on the **ISIC 2019 and ISIC 2020 dataset**. 
The goal is to develop a robust model leveraging a combination of **CNN, unsupervised and ensemble learning** for detecting skin lesions at the cancer stage as well as at pre-cancerous stages for early diagnosis and treatment.


---

## Table of Contents
- [Dataset](#datasets)
- [Additional Features](#additional-features)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

---

## Datasets

The dataset used in this project consists of both the **ISIC 2019** and **ISIC 2020** challenge datasets, which together contain a diverse set of dermoscopic images used for skin lesion classification.

### ISIC 2019 Dataset
The ISIC 2019 dataset contains **25,331 dermoscopic images** categorized into eight skin disease classes:

- **Melanoma (MEL)**
- **Melanocytic nevus (NV)**
- **Basal cell carcinoma (BCC)**
- **Actinic keratosis (AK)**
- **Benign keratosis (BKL)**
- **Dermatofibroma (DF)**
- **Vascular lesion (VASC)**
- **Squamous cell carcinoma (SCC)**

Cancerous lesions: Melanoma (most aggresive type of skin cancer), Basal cell carcinoma, Squamous cell carcinoma
Potential pre-cancerous lesions: Actinic keratosis

You can download the ISIC 2019 dataset from the [ISIC Challenge website](https://www.isic-archive.com).

### ISIC 2020 Dataset
The ISIC 2020 dataset contains **about 25,000 dermoscopic images** categorized into the following skin disease classes:

- **Melanoma**
- **Nevus**
- **Seborrheic keratosis**
- **Lentigo NOS**
- **Lichenoid keratosis**
- **Solar lentigo**
- **Cafe-au-lait macule**
- **atypical melanocytic proliferation**
- **Unknown**

Cancerous lesions: Melanoma (most aggresive type of skin cancer)
Potential pre-cancerous lesions: Lentigo NOS, Atypical melanocytic proliferation


You can download the ISIC 2020 dataset from the [ISIC Challenge website](https://www.isic-archive.com).

### Combined Usage

Both datasets are used to train the model for skin lesion classification. The ISIC 2020 dataset focuses on the melanomy cancerous lesion only while the ISIC 2019 dataset includes additional cancerous categories which can be leveraged for better generalization.
The ISIC 2020 dataset contains approximately 27000 images of "unknown" diagnosis, which were not used for model training. Only lesions with an established diagnosis were used in the final dataset.
Additionally, the ISIC 2019 dataset contains multiple images of the same lesions in different angles and lighting. While these provide valuable information we make sure that no data leakage of the same lesions across training and validation sets occurs.

---

### DL Frameworks 

This project uses Tensorflow for the data and CNN pipeline.

---

### Class Imbalance

Different techniques were implemented to handle the class imbalance of the combined dataset. 
- Oversampling of minority class with augmentations to create a balanced dataset. 
- Augmentations including rotations, random crop, brightness / contrast adjustments and coarse dropout. 

Focal loss and class weights were tested as alternatives to or in combination with oversampling but did not yield better results.

---

### Image cleaning

In order to make model learning easier, hairs were removed from all images by inpainting with OpenCV. To reduce computation time this was done 
separately before the images were used in the data pipeline.

The inpainting method also removed some of the image artefacts such as digital ruler markings from dermatoscopes. 

--- 

## Additional features
### Patient Metadata

Patient metadata was integrated with CNN features in the fully connected layers. 

### Pretraining with SimCLR Self-Supervised Learning

By pretraining on the training set images, SimCLR extracts high-level image representations that capture meaningful patterns and variations in the data. 
These learned features are then incorporated into the final model, along with patient metadata and CNN features. 

---

## Model Training
The model was trained by fine-tuning various deep learning architectures with pre-trained ImageNet weights to compare performance and test ensemble predictions.
- **ResNet-50** (used for pre-training only / self-supervised learning using SimCLR)
- **EfficientNet-B4**
- **DenseNet121**  
- **DenseNet169**
- **VGG-16**

For feature extraction the learning rate was set at 5e-5. In the fine-tuning phase a learning rate scheduler was used with a starting learning rate 5e-6 and exponential decay. 
Dropout and L2 regularization were used to prevent overfitting and increase generalisation. 

Image pre-processing was performed according to the requirements of each of the pre-trained models used.
The models were trained on AWS's g4dn.4xlarge instance.

---

## Evaluation

The optimal hyperparameters are found via randomised search and cross-validation on the training set. 

The model is evaluated based on:
- **F1 Score**
- **Precision & Recall**
- **AUC PR (Area Under the Precision-Recall Curve)**
- **AUC (ROC)**
- **Accuracy**  

---

## Results
** Ongoing / Preliminary results **

|    | Model              | AUC      | AUC PR   | F1 Score (thr. 0.5) | Precision (thr. 0.5) | Recall (thr. 0.5) | Accuracy (thr. 0.5) | F1 Score (opt. threshold) | Precision (opt. threshold) | Recall (opt. threshold) | Accuracy (opt. threshold) |
|----|--------------------|----------|----------|---------------------|----------------------|-------------------|---------------------|---------------------------|----------------------------|-------------------------|---------------------------|
| #1 | DenseNet121        | 0.92     | 0.85     | 0.77                | 0.76                 | 0.79              | 0.85                | **0.78**                  | **0.74**                   | **0.83**                | **0.85**                  |
| #2 | DenseNet169        | 0.92     | 0.83     | 0.75                | 0.79                 | 0.71              | 0.84                | 0.77                      | 0.71                       | 0.85                    | 0.83                      |
| #3 | EfficientNetB4     | 0.90     | 0.82     | 0.75                | 0.75                 | 0.74              | 0.83                | 0.75                      | 0.68                       | 0.85                    | 0.82                      |
| #4 | VGG16              | 0.90     | 0.81     | 0.75                | 0.70                 | 0.80              | 0.82                | 0.75                      | 0.68                       | 0.83                    | 0.82                      |
|    | Ensemble (#1 & 2)  | 0.92     | 0.85     |                     |                      |                   |                     | 0.78                      | 0.75                       | 0.81                    | 0.85                      |
|    | **Ensemble (all)** | **0.93** | **0.85** |                     |                      |                   |                     | **0.79**                  | **0.75**                   | **0.83**                | **0.85**                  |
|    |                    |          |          |                     |                      |                   |                     |                           |                            |                         |                           |
|    | Naive Classifier   | 0.5      | -        | 0.0                 | 0.0                  | 0.0               |                     |                           |                            |                         |                           |
|    | Random Classifier  | 0.5      | 0.34     | 0.32                | 0.32                 | 0.33              | 0.55                |                           |                            |                         |                           |
|    | Vanilla Classifier |          |          |                     |                      |                   |                     |                           |                            |                         |                           |
---

## License

This project uses the **ISIC 2019** and **ISIC 2020** Challenge Datasets, which are licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC 4.0)**.  

### **ISIC 2020 Dataset Attribution**  
To comply with the attribution requirements of this license, please cite the dataset as follows:  

**International Skin Imaging Collaboration**  
SIIM-ISIC 2020 Challenge Dataset. International Skin Imaging Collaboration.  
[DOI: 10.34970/2020-ds01](https://doi.org/10.34970/2020-ds01) (2020).  

🔗 **License:** [Creative Commons License](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt)  

The dataset was generated by the **International Skin Imaging Collaboration (ISIC)**, with images contributed by:  
- Hospital Clínic de Barcelona  
- Medical University of Vienna  
- Memorial Sloan Kettering Cancer Center  
- Melanoma Institute Australia  
- The University of Queensland  
- University of Athens Medical School  

#### **Citing ISIC 2020**  
> **Rotemberg, V., Kurtansky, N., Betz-Stablein, B., et al.**  
> *A patient-centric dataset of images and metadata for identifying melanomas using clinical context.*  
> Sci Data 8, 34 (2021). [DOI: 10.1038/s41597-021-00815-z](https://doi.org/10.1038/s41597-021-00815-z)  

### **ISIC 2019 Dataset Attribution**  
To comply with the attribution requirements of the **CC-BY-NC license**, the aggregate **"ISIC 2019: Training"** data must be cited as:  

- **BCN_20000 Dataset**: © Department of Dermatology, Hospital Clínic de Barcelona  
- **HAM10000 Dataset**: © ViDIR Group, Department of Dermatology, Medical University of Vienna  
  🔗 [DOI: 10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)  
- **MSK Dataset**: © Anonymous  
  🔗 [arXiv:1710.05006](https://arxiv.org/abs/1710.05006), [arXiv:1902.03368](https://arxiv.org/abs/1902.03368)  

#### **Citing ISIC 2019**  
> **Tschandl P., Rosendahl C., Kittler H.**  
> *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.*  
> Sci. Data 5, 180161 (2018). [DOI: 10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)  

> **Noel C. F. Codella, David Gutman, et al.**  
> *Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC).*  
> arXiv:1710.05006 (2017).  

> **Hernández-Pérez C., Combalia M., Podlipnik S., et al.**  
> *BCN20000: Dermoscopic lesions in the wild.*  
> Scientific Data 11, 641 (2024).  

---