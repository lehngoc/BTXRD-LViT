#!/bin/bash

python src/preprocessing/convert_labelme_to_mask.py
python src/preprocessing/crop_xray_border.py
python src/preprocessing/generate_text_annotations.py