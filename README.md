Since my paper is still in the publication process, only the CNN model code for the first dataset has been shared.

# Malaria Detection Using Deep Learning Models

This project involves a deep learning study aimed at automatically detecting malaria infection using three different cellular image datasets. A separate model was trained for each dataset, and the results were evaluated based on accuracy scores.


---

## 📁 Data sets

1. Dataset – Chittagong Medical College Hospital (Bangladesh)
<img width="482" height="695" alt="image" src="https://github.com/user-attachments/assets/7a030e5b-1aa0-44dc-9ec5-939542304b89" />

This dataset consists of microscope images from 150 confirmed malaria cases monitored at Chittagong Medical College Hospital in Bangladesh. Each image was annotated by medical experts and contains high-resolution cellular structures of both infected and non-infected samples. Since the images were collected in a clinical environment, the cell morphologies accurately represent real patient data.

2. Dataset – Trakya University Microbiology Laboratory
<img width="449" height="345" alt="image" src="https://github.com/user-attachments/assets/a80a138e-94fa-489f-908b-4107917d0320" />

This dataset contains microscope images prepared from various clinical blood samples by the Microbiology Department of Trakya University. Due to differences in staining techniques and laboratory conditions, the images exhibit variability in color intensity and cellular structure. This diversity enhances the model’s ability to generalize in real-world scenarios.

3. Hybrid Dataset – Combined Multisource Image Pool

The third dataset is a hybrid image pool created by merging the first two datasets. Since it includes microscope images obtained from two different geographic regions and two distinct laboratory environments, it provides broader variation, improved class balance, and a more robust foundation for achieving stronger generalization performance.

This study focuses on the detection of Plasmodium falciparum, one of the most severe species of malaria.



## 🧠 Models 

Three separate models were trained for each dataset, and their performance results were compared.

1. CNN (Convolutional Neural Network)

A classical convolutional neural network architecture offering a lightweight, fast, and low-parameter structure.

Trained on:

Dataset 1: Chittagong Medical College Hospital

Dataset 2: Trakya University Microbiology Laboratory

Dataset 3: Hybrid dataset

2. VGG-16 Wft

The VGG-16 model was experimented with, and fine-tuning was applied during training to improve feature extraction and classification performance.

Trained on:

Dataset 1: Chittagong Medical College Hospital

Dataset 2: Trakya University Microbiology Laboratory

Dataset 3: Hybrid dataset

3. CNN-ViT (Hybrid Model)

A hybrid architecture combining the strengths of CNNs and Vision Transformers to capture both local and global image representations more effectively.

Trained on:

Dataset 1: Chittagong Medical College Hospital

Dataset 2: Trakya University Microbiology Laboratory

Dataset 3: Hybrid dataset

## 📊 Results (Accuracy Only – Limited Due to Publication Process)

The highest accuracy result for each dataset is shared below.

Dataset 1:
The highest accuracy was achieved by the VGG-16 Wft model with 97.06%.

Dataset 2:
All three models achieved the same highest accuracy: 98.18%.

Dataset 3:
The highest accuracy was obtained with the VGG-16 Wft model, reaching 96.85%.

Note: Additional metrics (loss, precision, recall, F1-score, MCC, etc.) are not shared due to the ongoing publication process.
