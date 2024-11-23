SRC_FOLDER = $(shell pwd)
PREPROCESSING_FOLDER = $(SRC_FOLDER)/preprocessing
DATA_FOLDER = ../data
DATASET_FOLDER = $(DATA_FOLDER)/dataset
TINY_IMAGENET_FOLDER = $(DATA_FOLDER)/tiny-imagenet-200

PYTHONPATH = $(SRC_FOLDER)

.PHONY: download run clean

run:
	@echo "Running preprocessing..."
	python $(PREPROCESSING_FOLDER)/create_dataset.py 