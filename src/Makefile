N_TRAIN := 2000
N_TEST := 900
N_CLASSES := 10
MASK_PERCENTAGE := 5
RGB := False
BATCH_SIZE := 64
EPOCHS := 10
LEARNING_RATE := 0.001
IMAGES_WIDTH := 64
IMAGES_HEIGHT := 64
N_CHANNELS := 1
INITIALIZE := True
LR_SCHEDULER := True
STEP_SIZE := 1
GAMMA := LEARNING_RATE / EPOCHS

# simple_conv, conv_unet, conv_maxpool, transformer
MODEL_NAME := conv_maxpool

# cifar10, tinyimagenet, groceries
DATASET_NAME := tinyimagenet

ifeq ($(MODEL_NAME), simple_conv)
	MIDDLE_LAYERS := 64 128 256
	KERNEL_SIZES := 3 3 3
	STRIDES := 2 2 2
	PADDINGS := 1 1 1
	OUTPUT_PADDINGS := 1 1 1
else ifeq ($(MODEL_NAME), conv_unet)
	MIDDLE_LAYERS := 64 128 256
	KERNEL_SIZES := 3 3 3
	STRIDES := 1 1 1
	PADDINGS := 1 1 1
	OUTPUT_PADDINGS := 0 0 0
else ifeq ($(MODEL_NAME), conv_maxpool)
	MIDDLE_LAYERS := 64 128 256 512 1024
	KERNEL_SIZES := 3 3 3
	STRIDES := 1 1 1
	PADDINGS := 1 1 1
	OUTPUT_PADDINGS := 0 0 0
else ifeq ($(MODEL_NAME), transformer)
	PATCH_SIZE := 16
	EMBEDDING_DIM := 1024
	NUM_HEADS := 16
	NUM_LAYERS := 8
endif

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
	N_CHANNELS := 3
else
	TRAIN_PATH := $(TRAIN_PATH).pth
	TEST_PATH := $(TEST_PATH).pth
endif


LOSS_DATA_PATH := $(LOSSES_DATA_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).csv
LR_DATA_PATH := $(LR_DATA_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).csv
RESULTS_PLOT_PATH := $(RESULTS_PLOT_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).png
RESULTS_PATH := $(RESULTS_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).json
WEIGHTS_PATH := $(WEIGHTS_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).pth
CONFIG_FILE_PATHS := $(SRC_FOLDER)/paths.json
CONFIG_FILE_PARAMS := $(SRC_FOLDER)/params.json


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


	@echo '{"model_name": "$(MODEL_NAME)", "n_train": "$(N_TRAIN)", "n_test": "$(N_TEST)", "n_classes": "$(N_CLASSES)", "image_width": "$(IMAGES_WIDTH)", "image_height": "$(IMAGES_HEIGHT)", "n_channels": "$(N_CHANNELS)", "mask_percentage": "$(MASK_PERCENTAGE)", "rgb": "$(RGB)", "batch_size": "$(BATCH_SIZE)", "epochs": "$(EPOCHS)", "learning_rate": "$(LEARNING_RATE)", "scheduler": "$(LR_SCHEDULER)", "sched_step": "$(STEP_SIZE)", "sched_gamma": "$(GAMMA)", "initialize": "$(INITIALIZE)"' > $(CONFIG_FILE_PARAMS)

ifeq ($(MODEL_NAME), transformer)
	@echo ', "patch_size": "$(PATCH_SIZE)", "embedding_dim": "$(EMBEDDING_DIM)", "num_heads": "$(NUM_HEADS)", "num_layers": "$(NUM_LAYERS)"' >> $(CONFIG_FILE_PARAMS)
else
	@echo ', "middle_layers": "$(MIDDLE_LAYERS)", "kernel_sizes": "$(KERNEL_SIZES)", "strides": "$(STRIDES)", "paddings": "$(PADDINGS)", "output_paddings": "$(OUTPUT_PADDINGS)"' >> $(CONFIG_FILE_PARAMS)
endif

	@echo '}' >> $(CONFIG_FILE_PARAMS)

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
