BASE_MAKEFILE_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
BASE_DIR := $(realpath $(BASE_MAKEFILE_DIR)/../)

SRC_FOLDER := $(BASE_DIR)/src
PYTHONPATH := $(SRC_FOLDER)
PYTHON := $(shell which python3)

CONFIG_FILE_PATHS := $(SRC_FOLDER)/paths.json
CONFIG_FILE_PARAMS := $(SRC_FOLDER)/params.json

UNIX_TIMESTAMP := $(shell date +"%s")

N_TRAIN := $(shell jq -r '.dataset_params.n_train' $(CONFIG_FILE_PARAMS))
N_TEST := $(shell jq -r '.dataset_params.n_test' $(CONFIG_FILE_PARAMS))
N_CLASSES := $(shell jq -r '.dataset_params.n_classes' $(CONFIG_FILE_PARAMS))
RGB := $(shell jq -r '.dataset_params.rgb' $(CONFIG_FILE_PARAMS))
N_CHANNELS := $(shell jq -r '.dataset_params.n_channels' $(CONFIG_FILE_PARAMS))
DATASET_NAME := $(shell jq -r '.dataset_params.dataset_name' $(CONFIG_FILE_PARAMS))
MASK_TYPE := $(shell jq -r '.dataset_params.mask_name' $(CONFIG_FILE_PARAMS))
MASK_PARAMS := $(shell jq -r '.masks_configs' $(CONFIG_FILE_PARAMS))
REPEAT_IMAGES := $(shell jq -r '.dataset_params.repeat_images' $(CONFIG_FILE_PARAMS))

BATCH_SIZE := $(shell jq -r '.train_params.batch_size' $(CONFIG_FILE_PARAMS))
EPOCHS := $(shell jq -r '.train_params.epochs' $(CONFIG_FILE_PARAMS))
LEARNING_RATE := $(shell jq -r '.train_params.learning_rate' $(CONFIG_FILE_PARAMS))
INITIALIZE := $(shell jq -r '.train_params.initialize' $(CONFIG_FILE_PARAMS))
LR_SCHEDULER := $(shell jq -r '.train_params.scheduler' $(CONFIG_FILE_PARAMS))
MODEL_NAME := $(shell jq -r '.train_params.model_name' $(CONFIG_FILE_PARAMS))
MODEL_PARAMS := $(shell jq -r '.model_configs' $(CONFIG_FILE_PARAMS))

MODEL_IDENTIFIER := $(UNIX_TIMESTAMP)

DATASET_IDENTIFIER := $(N_CLASSES)_$(MASK_TYPE)
ifeq ($(REPEAT_IMAGES), True)
	DATASET_IDENTIFIER := $(DATASET_IDENTIFIER)_repeated
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
OPTIM_FOLDER := $(DATA_FOLDER)/optimization/$(DATASET_NAME)
TESTS_FOLDER := $(SRC_FOLDER)/tests
RESULTS_PLOT_FOLDER := $(FIGURES_FOLDER)/results
SAMPLES_FOLDER := $(FIGURES_FOLDER)/$(MODEL_IDENTIFIER)
SUMMARY_FOLDER := $(DATA_FOLDER)/summaries


DATASET_CREATION_CODE_PATH := $(PREPROCESSING_FOLDER)/create_dataset.py
TRAIN_CODE_PATH := $(SRC_FOLDER)/train.py

TRAIN_PATH := $(DATASET_FOLDER)/train/dataset_$(N_TRAIN)_$(DATASET_IDENTIFIER).pth
TEST_PATH := $(DATASET_FOLDER)/test/dataset_$(N_TEST)_$(DATASET_IDENTIFIER).pth

RESULTS_PLOT_PATH := $(RESULTS_PLOT_FOLDER)/$(MODEL_IDENTIFIER).png
RESULTS_PATH := $(RESULTS_FOLDER)/$(MODEL_IDENTIFIER).json
WEIGHTS_PATH := $(WEIGHTS_FOLDER)/$(MODEL_IDENTIFIER).pth
OPTIM_PATH := $(OPTIM_FOLDER)/$(MODEL_IDENTIFIER).json
SUMMARY_PATH := $(SUMMARY_FOLDER)/$(MODEL_IDENTIFIER).txt


.DEFAULT:
	@echo "Invalid target. Please run 'make help' for more information."

.PHONY: config preprocess train test zipw help

config:
	@echo "{" > $(CONFIG_FILE_PATHS)
	@echo '  "raw_data_folder": "$(RAW_DATA_FOLDER)",' >> $(CONFIG_FILE_PATHS)
	@echo '  "dataset_path": "$(DATASET_FOLDER)",' >> $(CONFIG_FILE_PATHS)
	@echo '  "weights_path": "$(WEIGHTS_PATH)",' >> $(CONFIG_FILE_PATHS)
	@echo '  "loss_figure_path": "$(LOSS_FIGURE_PATH)",' >> $(CONFIG_FILE_PATHS)
	@echo '  "results_path": "$(RESULTS_PATH)",' >> $(CONFIG_FILE_PATHS)
	@echo '  "results_plot_path": "$(RESULTS_PLOT_PATH)",' >> $(CONFIG_FILE_PATHS)
	@echo '  "samples_folder": "$(SAMPLES_FOLDER)/",' >> $(CONFIG_FILE_PATHS)
	@echo '  "optim_path": "$(OPTIM_PATH)",' >> $(CONFIG_FILE_PATHS)
	@echo '  "train_path": "$(TRAIN_PATH)", ' >> $(CONFIG_FILE_PATHS)
	@echo '  "test_path": "$(TEST_PATH)",' >> $(CONFIG_FILE_PATHS)
	@echo '  "summary_path": "$(SUMMARY_PATH)"' >> $(CONFIG_FILE_PATHS)
ifeq ($(DATASET_NAME), tinyimagenet)
	@echo ',' >> $(CONFIG_FILE_PATHS)
	@echo '  "image_extension": ".JPEG"' >> $(CONFIG_FILE_PATHS)
else ifeq ($(DATASET_NAME), groceries)
	@echo ',' >> $(CONFIG_FILE_PATHS)
	@echo ' "image_extension": ".png"' >> $(CONFIG_FILE_PATHS)
endif
	@echo "}" >> $(CONFIG_FILE_PATHS)

preprocess : config
	@echo "\nCreating train dataset with $(DATASET_CREATION_CODE_PATH) code..."
	$(PYTHON) $(DATASET_CREATION_CODE_PATH) --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)

create_summary:
	@echo "Creating summary..."
	@echo "Summary of the training process" > $(SUMMARY_PATH)
	@echo "Sample Identifier: $(MODEL_IDENTIFIER)" >> $(SUMMARY_PATH)
	@echo "" >> $(SUMMARY_PATH)
	@echo "Dataset Name: $(DATASET_NAME)" >> $(SUMMARY_PATH)
	@echo "Number of Training Samples: $(N_TRAIN)" >> $(SUMMARY_PATH)
	@echo "Number of Testing Samples: $(N_TEST)" >> $(SUMMARY_PATH)
	@echo "Number of Classes: $(N_CLASSES)" >> $(SUMMARY_PATH)
	@echo "Repeat Images: $(REPEAT_IMAGES)" >> $(SUMMARY_PATH)
	@echo "RGB: $(RGB)" >> $(SUMMARY_PATH)
	@echo "Mask type: $(MASK_TYPE)" >> $(SUMMARY_PATH)
	@echo "Mask Parameters: $(MASK_PARAMS)" >> $(SUMMARY_PATH)
	@echo "" >> $(SUMMARY_PATH)
	@echo "Model Parameters" >> $(SUMMARY_PATH)
	@echo "Model Name: $(MODEL_NAME)" >> $(SUMMARY_PATH)
	@echo "Model parameters: $(MODEL_PARAMS)" >> $(SUMMARY_PATH)
	@echo "" >> $(SUMMARY_PATH)
	@echo "Training Parameters" >> $(SUMMARY_PATH)
	@echo "Batch Size: $(BATCH_SIZE)" >> $(SUMMARY_PATH)
	@echo "Epochs: $(EPOCHS)" >> $(SUMMARY_PATH)
	@echo "Learning Rate: $(LEARNING_RATE)" >> $(SUMMARY_PATH)
	@echo "Scheduler: $(LR_SCHEDULER)" >> $(SUMMARY_PATH)
	@echo "Initialize: $(INITIALIZE)" >> $(SUMMARY_PATH)

train: create_summary config
	@echo "Training the model $(MODEL_IDENTIFIER)..."
	$(PYTHON) $(TRAIN_CODE_PATH) --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)

plot: config
	@echo "Plotting the results..."
	$(PYTHON) $(SRC_FOLDER)/plot_results.py --results_folder $(RESULTS_FOLDER) --figs_folder $(RESULTS_PLOT_FOLDER)

optimize: config
	@echo "Optimizing the model..."
	$(PYTHON) $(SRC_FOLDER)/bayesian_optimization.py --paths $(CONFIG_FILE_PATHS)

test: config
	@echo "Running test..."
	PYTHONPATH=$(PYTHONPATH) pytest

zipw:
	@echo "Zipping the weights..."
	$(PYTHON) $(SRC_FOLDER)/zip_weights.py --weights_folder $(WEIGHTS_FOLDER)

help:
	@echo "Makefile targets:"
	@echo "  config      - Generate configuration files"
	@echo "  preprocess  - Preprocess the dataset"
	@echo "  train       - Train the model"
	@echo "  test        - Run tests"
	@echo "  plot        - Plot the results from the json files"
	@echo "  zip		 - Zip the json weights"
	@echo "  help        - Display this help message"
