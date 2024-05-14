# pokemon
EM segmentation group project for DL4MIA

## Setup

Create an environment with:

    mamba create -n pokemon -c pytorch -c nvidia -c conda-forge --file requirements.txt -y


### EM data from Chad

The data (from Chad) can be found in `/group/dl4miacourse/pokemon/data-chad/First40training_last10validation/`

    /group/dl4miacourse/pokemon/data-chad/First40training_last10validation/
    ├── First40_training_annotations
    ├── First40_training_images
    ├── Last10_validation_annotations
    └── Last10_validation_images

The images are PNG files and the annotations are in a single JSON file. Each annotation box is a polygon with alternating XY coordinates.

### Actual usage

1. Change the `organelle` variable in `train.py` and run `python train.py`
2. Change the `organelle` variable in `evaluate.py` and run `python evaluate.py`