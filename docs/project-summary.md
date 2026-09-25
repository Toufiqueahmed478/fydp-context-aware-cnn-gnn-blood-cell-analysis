# Project summary

## Official project identity

- **Final title:** Context-Aware Graph Neural Network for Blood Cell Morphology and Disease Analysis
- **Project type:** Final Year Design Project
- **Department:** Department of Computer Systems Engineering
- **University:** Quaid-e-Awam University of Engineering, Science and Technology, Nawabshah
- **Academic year:** 2026

## Project team

- Sadia Bhutto - Group Leader
- Toufique Ahmed
- Zulfiqar Ali Bhutto

## Supervision

- **Academic supervisor:** Engr. Iftikhar Ahmed Koondhar, Assistant Professor
- **Industrial supervisor:** Dr. Mashal Memon, Pediatrician, Institute of Maternal and Child Health

## Problem and aim

Manual blood-cell analysis is time-consuming and depends heavily on specialist expertise. This project develops a student research prototype for classifying individual blood cells from microscopy images and providing a preliminary, rule-based hematological screening interpretation.

## Methodology

1. Extract individual cells from YOLO-annotated microscopy images.
2. Resize and normalize the cropped cell images.
3. Address class imbalance by creating a training set with 3,000 samples per class through controlled undersampling and augmentation.
4. Build a three-node morphology graph representing the whole cell, nucleus, and cytoplasm.
5. Extract 12 handcrafted morphology and colour features for each graph node.
6. Train and compare an improved GNN baseline, a pretrained ResNet18 CNN baseline, and a hybrid CNN-GNN model.
7. Evaluate on both the naturally imbalanced original test set and a balanced fair test set.
8. Present predictions, confidence values, screening explanations, and a PDF report through a Gradio interface.

## Dataset source

The project uses the [OI-PBC Dataset: 298,850 blood cells and 12 classes](https://www.kaggle.com/datasets/mohamadabouali1/oi-pbc-dataset-298850-blood-cells-and-12-classes) hosted on Kaggle. The raw dataset is not redistributed in this repository.

## Cell classes

The final implementation uses 12 classes: BA, BNE, EO, ERB, LY, MMY, MO, MY, PLT, PMY, RBC, and SNE.

## Hybrid model

The CNN branch uses a pretrained ResNet18 and produces a 512-dimensional visual representation. The graph branch uses GCN layers with 12-dimensional node features and produces a 96-dimensional graph representation. These representations are concatenated before the final classification head.

## Reported results

The submitted thesis reports the following results for the hybrid model:

- Original test set: 99.35% accuracy, 90.92% balanced accuracy, 90.96% macro-F1, Cohen's kappa 0.966, and ROC-AUC 0.990.
- Balanced fair test set: 91.15% accuracy and 91.17% macro-F1, with MCC 0.911 and ROC-AUC 0.990.

These values are reported for the OI-PBC/PBC-YOLO-721 data used in the project. They are not evidence of clinical performance on other hospitals, devices, staining protocols, or patient populations. The corrected notebook must reproduce the final values before public release.

## Disease-level interpretation

The current interface maps individual predicted cell classes to preliminary screening categories such as anemia-related, leukemia-related, abnormal myeloid maturation, and platelet-related analysis. This is a rule-based interpretation layer, not a separately trained disease-diagnosis model and not a clinical diagnosis.

## Limitations

- The current model operates on cropped single-cell images rather than complete blood-smear slides.
- The model was tested on the project dataset only; external generalization was not established.
- Rare classes may have less stable performance because the original sample counts are small.
- Population-level disease assessment from multiple cells and laboratory values is not implemented.
- The dashboard is a research prototype and has not been clinically validated or certified.

## Future work

Planned extensions include whole-slide detection, Graph Attention Networks, multi-cell disease-level aggregation, external validation, explainable AI, and clinical collaboration.

## Source documents

The original proposal and thesis were used to prepare this summary. Their full PDFs contain personal contact details, roll numbers, signatures, and university submission material. They should not be added to a public repository without approval from all project members and supervisors and without removing information that should remain private.
