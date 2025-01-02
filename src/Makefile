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

MODEL_IDENTIFIER := $(N_TRAIN)_$(N_TEST)_$(N_CLASSES)_$(MASK_PERCENTAGE)_$(EPOCHS)_$(BATCH_SIZE)


BASE_MAKEFILE_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
BASE_DIR := $(realpath $(BASE_MAKEFILE_DIR)/../)

SRC_FOLDER := $(BASE_DIR)/src
PYTHONPATH := $(SRC_FOLDER)
PYTHON := $(shell which python3)
MODEL_NAME := transformer

DATA_FOLDER := $(BASE_DIR)/data
FIGURES_FOLDER := $(BASE_DIR)/figs
LOSSES_FOLDER := $(FIGURES_FOLDER)/losses
MODELS_FOLDER := $(SRC_FOLDER)/models
PREPROCESSING_FOLDER := $(SRC_FOLDER)/data_preprocessing

CODE_FOLDER := $(MODELS_FOLDER)
DATASET_FOLDER := $(DATA_FOLDER)/datasets
TINY_IMAGENET_FOLDER := $(DATA_FOLDER)/tiny-imagenet-200/train
WEIGTHS_FOLDER := $(DATA_FOLDER)/weights

TRAIN_PATH := $(DATASET_FOLDER)/train/dataset_$(N_TRAIN)_$(N_CLASSES)_$(MASK_PERCENTAGE).pth
TEST_PATH := $(DATASET_FOLDER)/test/dataset_$(N_TEST)_$(N_CLASSES)_$(MASK_PERCENTAGE).pth
TRAIN_CODE_PATH := $(CODE_FOLDER)/train.py
WEIGTHS_PATH := $(WEIGTHS_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).pth
LOSS_FIGURE_PATH := $(LOSSES_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).png
CONFIG_FILE_PATHS := $(SRC_FOLDER)/paths.json
CONFIG_FILE_PARAMS := $(SRC_FOLDER)/params.json


.DEFAULT:
	@echo "Invalid target. Please run 'make help' for more information."

.PHONY: config PREPROCESSING_FOLDER

config:
	@echo '{"tiny_imagenet_folder": "$(TINY_IMAGENET_FOLDER)", "train_path": "$(TRAIN_PATH)", "test_path": "$(TEST_PATH)", "weights_path": "$(WEIGTHS_PATH)", "loss_figure_path": "$(LOSS_FIGURE_PATH)"}' > $(CONFIG_FILE_PATHS)
	@echo '{"n_train": $(N_TRAIN), "n_test": $(N_TEST), "n_classes": $(N_CLASSES), "image_width": $(IMAGES_WIDTH), "image_height": $(IMAGES_HEIGHT), "n_channels": $(N_CHANNELS), "mask_percentage": $(MASK_PERCENTAGE), "batch_size": $(BATCH_SIZE), "epochs": $(EPOCHS), "learning_rate": $(LEARNING_RATE)}' > $(CONFIG_FILE_PARAMS)

preprocess : config
	@echo "Creating train dataset..."
	$(PYTHON) $(PREPROCESSING_FOLDER)/create_dataset.py --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)

train: config
	$(PYTHON) $(TRAIN_CODE_PATH) --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)
