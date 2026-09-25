# Dataset instructions

The project uses the [OI-PBC blood-cell dataset on Kaggle](https://www.kaggle.com/datasets/mohamadabouali1/oi-pbc-dataset-298850-blood-cells-and-12-classes) with YOLO annotations. The implemented notebook selects the PBC-YOLO-721 dataset version and extracts 259,590 valid cell crops from the train, validation, and test splits. The raw dataset is not included in this repository.

Before publication, confirm the Kaggle dataset's current licence and add the exact attribution required by its owner:

1. The official dataset name and source URL.
2. The dataset licence or usage conditions.
3. The expected folder structure.
4. Instructions for placing the dataset locally or in Google Drive.
5. A note explaining whether the dataset is based on PBC-YOLO-721 or another version.

The final implementation contains 12 classes and uses an original imbalanced test set plus a balanced fair test subset. The proposal refers to 11 classes and Streamlit in some sections; those statements describe an earlier plan and should not be copied into the final README.

Do not upload private, restricted, or unlicensed data.
