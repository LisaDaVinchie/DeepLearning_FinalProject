N_TRAIN := 5000
N_TEST := 2000
N_CLASSES := 20
MASK_PERCENTAGE := 5
BATCH_SIZE := 64
EPOCHS := 10
LEARNING_RATE := 0.001
IMAGES_WIDTH := 64
IMAGES_HEIGHT := 64
N_CHANNELS := 3
INITIALIZE := True
LR_SCHEDULER := True
STEP_SIZE := 1
GAMMA := 0.1
MODEL_NAME := transformer

MODEL_IDENTIFIER := $(N_TRAIN)_$(N_TEST)_$(N_CLASSES)_$(MASK_PERCENTAGE)_$(EPOCHS)_$(BATCH_SIZE)_$(LEARNING_RATE)

ifeq ($(LR_SCHEDULER), True)
	MODEL_IDENTIFIER := $(MODEL_IDENTIFIER)_scheduled
endif

ifeq ($(INITIALIZE), True)
	MODEL_IDENTIFIER := $(MODEL_IDENTIFIER)_initialized
endif


BASE_MAKEFILE_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
BASE_DIR := $(realpath $(BASE_MAKEFILE_DIR)/../)

SRC_FOLDER := $(BASE_DIR)/src
PYTHONPATH := $(SRC_FOLDER)
PYTHON := $(shell which python3)

DATA_FOLDER := $(BASE_DIR)/data
FIGURES_FOLDER := $(BASE_DIR)/figs
LOSSES_FOLDER := $(FIGURES_FOLDER)/losses
MODELS_FOLDER := $(SRC_FOLDER)/models
PREPROCESSING_FOLDER := $(SRC_FOLDER)/data_preprocessing

CODE_FOLDER := $(MODELS_FOLDER)
DATASET_FOLDER := $(DATA_FOLDER)/datasets
TINY_IMAGENET_FOLDER := $(DATA_FOLDER)/tiny-imagenet-200/train
WEIGTHS_FOLDER := $(DATA_FOLDER)/weights
LOSSES_DATA_FOLDER := $(DATA_FOLDER)/losses
LR_DATA_FOLDER := $(DATA_FOLDER)/learning_rates

TRAIN_PATH := $(DATASET_FOLDER)/train/dataset_$(N_TRAIN)_$(N_CLASSES)_$(MASK_PERCENTAGE).pth
TEST_PATH := $(DATASET_FOLDER)/test/dataset_$(N_TEST)_$(N_CLASSES)_$(MASK_PERCENTAGE).pth
LOSS_DATA_PATH := $(LOSSES_DATA_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).csv
LR_DATA_PATH := $(LR_DATA_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).csv
TRAIN_CODE_PATH := $(SRC_FOLDER)/train.py
WEIGTHS_PATH := $(WEIGTHS_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).pth
LOSS_FIGURE_PATH := $(LOSSES_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).png
CONFIG_FILE_PATHS := $(SRC_FOLDER)/paths.json
CONFIG_FILE_PARAMS := $(SRC_FOLDER)/params.json


.DEFAULT:
	@echo "Invalid target. Please run 'make help' for more information."

.PHONY: config preprocess train

config:
	@echo '{"tiny_imagenet_folder": "$(TINY_IMAGENET_FOLDER)", "train_path": "$(TRAIN_PATH)", "test_path": "$(TEST_PATH)", "weights_path": "$(WEIGTHS_PATH)", "loss_figure_path": "$(LOSS_FIGURE_PATH)", "loss_data_path": "$(LOSS_DATA_PATH)", "learning_rates_path": "$(LR_DATA_PATH)"}' > $(CONFIG_FILE_PATHS)
	@echo '{"model_name": $(MODEL_NAME), "n_train": $(N_TRAIN), "n_test": $(N_TEST), "n_classes": $(N_CLASSES), "image_width": $(IMAGES_WIDTH), "image_height": $(IMAGES_HEIGHT), "n_channels": $(N_CHANNELS), "mask_percentage": $(MASK_PERCENTAGE), "batch_size": $(BATCH_SIZE), "epochs": $(EPOCHS), "learning_rate": $(LEARNING_RATE), "scheduler": "$(LR_SCHEDULER)", "sched_step": "$(STEP_SIZE)", "sched_gamma": "$(GAMMA)", "initialize": "$(INITIALIZE)"}' > $(CONFIG_FILE_PARAMS)

preprocess : config
	@echo "Creating train dataset..."
	$(PYTHON) $(PREPROCESSING_FOLDER)/create_dataset.py --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)

train: config
	@echo "Training the model..."
	$(PYTHON) $(TRAIN_CODE_PATH) --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)
