N_TRAIN := 10000
N_TEST := 4000
N_CLASSES := 50
MASK_PERCENTAGE := 5
BATCH_SIZE := 64
EPOCHS := 10
LEARNING_RATE := 0.001
IMAGES_WIDTH := 64
IMAGES_HEIGHT := 64
N_CHANNELS := 3


BASE_MAKEFILE_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
BASE_DIR := $(realpath $(BASE_MAKEFILE_DIR)/../)

SRC_FOLDER := $(BASE_DIR)/src
PYTHONPATH := $(SRC_FOLDER)
PYTHON := python3
MODEL_NAME := transformer

DATA_FOLDER := $(BASE_DIR)/data
DATASET_FOLDER := $(DATA_FOLDER)/datasets
TINY_IMAGENET_FOLDER := $(DATA_FOLDER)/tiny-imagenet-200/train
WEIGTHS_FOLDER := $(DATA_FOLDER)/weights

FIGURES_FOLDER := $(BASE_DIR)/figs
LOSSES_FOLDER := $(FIGURES_FOLDER)/losses

MODELS_FOLDER := $(SRC_FOLDER)/models/$(MODEL_NAME)
PREPROCESSING_FOLDER := $(SRC_FOLDER)/data_preprocessing

TRAIN_PATH := $(DATASET_FOLDER)/train/dataset_$(N_TRAIN)_$(N_CLASSES)_$(MASK_PERCENTAGE).pth
TEST_PATH := $(DATASET_FOLDER)/test/dataset_$(N_TEST)_$(N_CLASSES)_$(MASK_PERCENTAGE).pth


CONFIG_FILE_PATHS := $(SRC_FOLDER)/paths.json
CONFIG_FILE_PARAMS := $(SRC_FOLDER)/params.json


.DEFAULT:
	@echo "Invalid target. Please run 'make help' for more information."

.PHONY: config PREPROCESSING_FOLDER

config:
	@echo '{\
		"tiny_imagenet_folder": "$(TINY_IMAGENET_FOLDER)", \
		"train_path": "$(TRAIN_PATH)", \
		"test_path": "$(TEST_PATH)", \
		"weights_folder": "$(WEIGTHS_FOLDER)", \
		"losses_folder": "$(LOSSES_FOLDER)"}' | $(PYTHON) -m json.tool > $(CONFIG_FILE_PATHS)
	@echo '{\
		"n_train": $(N_TRAIN), \
		"n_test": $(N_TEST), \
		"n_classes": $(N_CLASSES), \
		"image_width": $(IMAGES_WIDTH), \
		"image_height": $(IMAGES_HEIGHT), \
		"n_channels": $(N_CHANNELS), \
		"mask_percentage": $(MASK_PERCENTAGE), \
		"batch_size": $(BATCH_SIZE), \
		"epochs": $(EPOCHS), \
		"learning_rate": $(LEARNING_RATE)}' | $(PYTHON) -m json.tool > $(CONFIG_FILE_PARAMS)

preprocess : config
	@echo "Creating train dataset..."
	$(PYTHON) $(PREPROCESSING_FOLDER)/create_dataset.py --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)