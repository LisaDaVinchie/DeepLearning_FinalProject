BASE_MAKEFILE_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
BASE_DIR := $(realpath $(BASE_MAKEFILE_DIR)/../)

SRC_FOLDER := $(BASE_DIR)/src
PYTHONPATH := $(SRC_FOLDER)
PYTHON := $(shell which python3)

CONFIG_FILE_PATHS := $(SRC_FOLDER)/paths.json
CONFIG_FILE_PARAMS := $(SRC_FOLDER)/params.json

N_TRAIN := $(shell jq -r '.n_train' $(CONFIG_FILE_PARAMS))
N_TEST := $(shell jq -r '.n_test' $(CONFIG_FILE_PARAMS))
N_CLASSES := $(shell jq -r '.n_classes' $(CONFIG_FILE_PARAMS))
MASK_PERCENTAGE := $(shell jq -r '.mask_percentage' $(CONFIG_FILE_PARAMS))
RGB := $(shell jq -r '.rgb' $(CONFIG_FILE_PARAMS))
BATCH_SIZE := $(shell jq -r '.batch_size' $(CONFIG_FILE_PARAMS))
EPOCHS := $(shell jq -r '.epochs' $(CONFIG_FILE_PARAMS))
LEARNING_RATE := $(shell jq -r '.learning_rate' $(CONFIG_FILE_PARAMS))
N_CHANNELS := $(shell jq -r '.n_channels' $(CONFIG_FILE_PARAMS))
INITIALIZE := $(shell jq -r '.initialize' $(CONFIG_FILE_PARAMS))
LR_SCHEDULER := $(shell jq -r '.scheduler' $(CONFIG_FILE_PARAMS))
MODEL_NAME := $(shell jq -r '.model_name' $(CONFIG_FILE_PARAMS))
DATASET_NAME := $(shell jq -r '.dataset_name' $(CONFIG_FILE_PARAMS))

MODEL_IDENTIFIER := $(N_TRAIN)_$(N_TEST)_$(N_CLASSES)_$(MASK_PERCENTAGE)_$(EPOCHS)_$(BATCH_SIZE)_$(LEARNING_RATE)

ifeq ($(LR_SCHEDULER), True)
	MODEL_IDENTIFIER := $(MODEL_IDENTIFIER)_scheduled
endif

ifeq ($(INITIALIZE), True)
	MODEL_IDENTIFIER := $(MODEL_IDENTIFIER)_initialized
endif

DATA_FOLDER := $(BASE_DIR)/data
FIGURES_FOLDER := $(BASE_DIR)/figs
LOSSES_FOLDER := $(FIGURES_FOLDER)/losses
MODELS_FOLDER := $(SRC_FOLDER)/models
PREPROCESSING_FOLDER := $(SRC_FOLDER)/data_preprocessing

CODE_FOLDER := $(MODELS_FOLDER)
RAW_DATA_FOLDER := $(DATA_FOLDER)/$(DATASET_NAME)
ifeq ($(DATASET_NAME), tinyimagenet)
	RAW_DATA_FOLDER := $(RAW_DATA_FOLDER)/train
endif

DATASET_FOLDER := $(DATA_FOLDER)/datasets/$(DATASET_NAME)
FIGURES_FOLDER := $(BASE_DIR)/figs/$(DATASET_NAME)

WEIGHTS_FOLDER := $(DATA_FOLDER)/weights/$(DATASET_NAME)
RESULTS_FOLDER := $(DATA_FOLDER)/results/$(DATASET_NAME)
LOSSES_DATA_FOLDER := $(DATA_FOLDER)/losses/$(DATASET_NAME)
TESTS_FOLDER := $(SRC_FOLDER)/tests
RESULTS_PLOT_FOLDER := $(FIGURES_FOLDER)/results


DATASET_CREATION_CODE_PATH := $(PREPROCESSING_FOLDER)/create_dataset.py
TRAIN_CODE_PATH := $(SRC_FOLDER)/train.py

TRAIN_PATH := $(DATASET_FOLDER)/train/dataset_$(N_TRAIN)_$(N_CLASSES)_$(MASK_PERCENTAGE)
TEST_PATH := $(DATASET_FOLDER)/test/dataset_$(N_TEST)_$(N_CLASSES)_$(MASK_PERCENTAGE)
ifeq ($(RGB), True)
	TRAIN_PATH := $(TRAIN_PATH)_rgb.pth
	TEST_PATH := $(TEST_PATH)_rgb.pth
else
	TRAIN_PATH := $(TRAIN_PATH).pth
	TEST_PATH := $(TEST_PATH).pth
endif


LOSS_DATA_PATH := $(LOSSES_DATA_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).csv
LR_DATA_PATH := $(LR_DATA_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).csv
RESULTS_PLOT_PATH := $(RESULTS_PLOT_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).png
RESULTS_PATH := $(RESULTS_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).json
WEIGHTS_PATH := $(WEIGHTS_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).pth


.DEFAULT:
	@echo "Invalid target. Please run 'make help' for more information."

.PHONY: config preprocess train test help

config:
	@echo '{"raw_data_folder": "$(RAW_DATA_FOLDER)", "dataset_path": "$(DATASET_FOLDER)", "weights_path": "$(WEIGHTS_PATH)", "loss_figure_path": "$(LOSS_FIGURE_PATH)", "results_path": "$(RESULTS_PATH)", "results_plot_path": "$(RESULTS_PLOT_PATH)"' > $(CONFIG_FILE_PATHS)
ifeq ($(DATASET_NAME), tinyimagenet)
	@echo ', "train_path": "$(TRAIN_PATH)", "test_path": "$(TEST_PATH)", "image_extension": ".JPEG"' >> $(CONFIG_FILE_PATHS)
else ifeq ($(DATASET_NAME), groceries)
	@echo ', "train_path": "$(TRAIN_PATH)", "test_path": "$(TEST_PATH)", "image_extension": ".png"' >> $(CONFIG_FILE_PATHS)
endif
	@echo '}' >> $(CONFIG_FILE_PATHS)

preprocess : config
	@echo "\nCreating train dataset with $(DATASET_CREATION_CODE_PATH) code..."
	$(PYTHON) $(DATASET_CREATION_CODE_PATH) --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)

train: config
	@echo "Training the model..."
	$(PYTHON) $(TRAIN_CODE_PATH) --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)

plot: config
	@echo "Plotting the results..."
	$(PYTHON) $(SRC_FOLDER)/plot_results.py --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)

optimize: config
	@echo "Optimizing the model..."
	$(PYTHON) $(SRC_FOLDER)/hyperparameter_optimization.py --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)

test: config
	@echo "Running test..."
	PYTHONPATH=$(PYTHONPATH) pytest

help:
	@echo "Makefile targets:"
	@echo "  config      - Generate configuration files"
	@echo "  preprocess  - Preprocess the dataset"
	@echo "  train       - Train the model"
	@echo "  test        - Run tests"
	@echo "  help        - Display this help message"
