# Context-Aware Graph Neural Network for Blood Cell Morphology and Disease Analysis

> **Important final-year-project disclaimer:** This is a student-developed Final Year Design Project prototype created for academic and educational purposes. It has not been clinically validated, certified, or approved for medical use. It must not be used to diagnose, screen, monitor, or treat any disease or to make medical decisions. Anyone who uses, modifies, or relies on this project, its models, predictions, explanations, or reports does so at their own risk. The student authors, supervisors, university, and contributors accept no responsibility or liability for any loss, harm, error, or decision resulting from its use or misuse. Consult qualified medical professionals for health-related decisions.

This repository is the review package for a Final Year Design Project (FYDP) on automated blood-cell classification using a hybrid Convolutional Neural Network (CNN) and Graph Neural Network (GNN). The completed project title is **Context-Aware Graph Neural Network for Blood Cell Morphology and Disease Analysis**.

Suggested public repository name: `fydp-context-aware-cnn-gnn-blood-cell-analysis`

## Project overview

The project classifies 12 blood-cell categories from cropped microscopy images. The workflow includes YOLO-based cell cropping, class balancing with training-only augmentation, morphology and colour graph features, CNN and GNN baselines, a hybrid CNN–GNN model, evaluation reports, and a Gradio dashboard.

The project was developed by Sadia Bhutto, Toufique Ahmed, and Zulfiqar Ali Bhutto in the Department of Computer Systems Engineering at Quaid-e-Awam University of Engineering, Science and Technology, Nawabshah. The academic supervisor was Engr. Iftikhar Ahmed Koondhar, and the industrial supervisor was Dr. Mashal Memon.

## Current review status

This package contains the current Colab notebook and the proposed GitHub structure. The notebook still needs one final cleanup pass before publication:

- The final evaluation cell currently stops with `NameError: name 'DEVICE' is not defined` in the saved run.
- Google Drive paths should be replaced with configurable local or Colab paths.
- The dashboard code should be extracted from the notebook into `app/app.py`.
- Final test metrics must be regenerated after the evaluation fix.

The submitted thesis reports a hybrid-model accuracy of 99.35% and macro-F1 of 90.96% on the original test set, and accuracy of 91.15% and macro-F1 of 91.17% on the balanced fair test set. These values should be reproduced from the corrected notebook before they are presented as the final repository results.

## Models

- Improved GNN baseline using morphology and colour graph features
- CNN baseline using ResNet18
- Proposed hybrid CNN + GNN classifier

## Dataset

The notebook currently expects the PBC-YOLO-721 dataset under a Google Drive folder. The complete dataset is intentionally not included in this repository. The dataset source is the [OI-PBC Dataset on Kaggle](https://www.kaggle.com/datasets/mohamadabouali1/oi-pbc-dataset-298850-blood-cells-and-12-classes). Dataset usage conditions and download instructions are documented in `data/README.md`.

## Project documentation

See [`docs/project-summary.md`](docs/project-summary.md) for the project scope, methodology, reported results, limitations, and future work extracted from the proposal and thesis.

## Repository plan

```text
notebooks/   Reproducible Colab notebook
src/         Training, evaluation, and inference code
app/         Gradio dashboard
configs/     Paths and experiment settings
results/     Selected figures and metrics
data/        Dataset instructions only
models/      Model download instructions only
docs/        Architecture and project documentation
```

## Intended use

This project is intended only for academic research and demonstration. The full disclaimer is available in [`docs/DISCLAIMER.md`](docs/DISCLAIMER.md).

## Authors

- Sadia Bhutto
- Toufique Ahmed
- Zulfiqar Ali Bhutto

## License

The code licence and dataset licence will be stated separately after the project ownership and dataset source are confirmed.
