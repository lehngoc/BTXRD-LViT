python src\preprocessing\audit_btxrd_dataset.py
python src\preprocessing\convert_labelme_to_mask.py
python src\preprocessing\visualize_masks.py
python src\preprocessing\preprocess_btxrd_data.py --clean-output
python src\preprocessing\validate_preprocessed_data.py
python src\preprocessing\visualize_preprocessed_data.py
python src\preprocessing\make_splits.py