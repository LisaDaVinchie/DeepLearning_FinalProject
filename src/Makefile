BASE_MAKEFILE_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
BASE_DIR := $(realpath $(BASE_MAKEFILE_DIR)/../)

SRC_FOLDER := $(BASE_DIR)/src
PYTHONPATH := $(SRC_FOLDER)
PYTHON := $(shell which python3)

CONFIG_FILE_PATHS := $(SRC_FOLDER)/paths.json
CONFIG_FILE_PARAMS := $(SRC_FOLDER)/params_thesis.json

N_TRAIN := $(shell jq -r '.dataset_params.n_train' $(CONFIG_FILE_PARAMS))
N_TEST := $(shell jq -r '.dataset_params.n_test' $(CONFIG_FILE_PARAMS))

BATCH_SIZE := $(shell jq -r '.train_params.batch_size' $(CONFIG_FILE_PARAMS))
EPOCHS := $(shell jq -r '.train_params.epochs' $(CONFIG_FILE_PARAMS))
LEARNING_RATE := $(shell jq -r '.train_params.learning_rate' $(CONFIG_FILE_PARAMS))
INITIALIZE := $(shell jq -r '.train_params.initialize' $(CONFIG_FILE_PARAMS))
LR_SCHEDULER := $(shell jq -r '.train_params.scheduler' $(CONFIG_FILE_PARAMS))
MODEL_NAME := $(shell jq -r '.train_params.model_name' $(CONFIG_FILE_PARAMS))

MODEL_IDENTIFIER := $(N_TRAIN)_$(N_TEST)_$(EPOCHS)_$(BATCH_SIZE)_$(LEARNING_RATE)

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
RAW_DATA_FOLDER := $(DATA_FOLDER)

BATHYMETRY_FOLDER := $(RAW_DATA_FOLDER)/bathymetry/EMODnet_bathymetry
LAND_FOLDER := $(RAW_DATA_FOLDER)/land/Copernicus_land_ortho
INTERPOLATED_FOLDER := $(RAW_DATA_FOLDER)/interpolated
WEIGHTS_FOLDER := $(DATA_FOLDER)/weights_thesis
RESULTS_FOLDER := $(DATA_FOLDER)/results_thesis
DATASET_FOLDER := $(DATA_FOLDER)/dataset

FIGURES_FOLDER := $(BASE_DIR)/figs_thesis

OPTIM_FOLDER := $(DATA_FOLDER)/optimization_thesis
TESTS_FOLDER := $(SRC_FOLDER)/tests
RESULTS_PLOT_FOLDER := $(FIGURES_FOLDER)/results_thesis
SAMPLES_FOLDER := $(FIGURES_FOLDER)/samples_thesis


DATASET_CREATION_CODE_PATH := $(PREPROCESSING_FOLDER)/create_dataset.py
TRAIN_CODE_PATH := $(SRC_FOLDER)/train.py

TRAIN_PATH := $(DATASET_FOLDER)/train/dataset_$(N_TRAIN)_$(DATASET_IDENTIFIER).pth
TEST_PATH := $(DATASET_FOLDER)/test/dataset_$(N_TEST)_$(DATASET_IDENTIFIER).pth

RESULTS_PLOT_PATH := $(RESULTS_PLOT_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).png
RESULTS_PATH := $(RESULTS_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).json
WEIGHTS_PATH := $(WEIGHTS_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).pth
OPTIM_PATH := $(OPTIM_FOLDER)/$(MODEL_NAME)_$(MODEL_IDENTIFIER).json


.DEFAULT:
	@echo "Invalid target. Please run 'make help' for more information."

.PHONY: config preprocess regrid train test zipw help

config:
	@echo '{' > $(CONFIG_FILE_PATHS)
	@echo '  "bathymetry" : {' >> $(CONFIG_FILE_PATHS)
	@echo '		"folder" : "$(BATHYMETRY_FOLDER)",' >> $(CONFIG_FILE_PATHS)
	@echo '		"base" : "$(BATHYMETRY_FOLDER)/base/",' >> $(CONFIG_FILE_PATHS)
	@echo '     "netcdf" : "$(BATHYMETRY_FOLDER)/netcdf/",' >> $(CONFIG_FILE_PATHS)
	@echo '     "numpy" : "$(BATHYMETRY_FOLDER)/npy/",' >> $(CONFIG_FILE_PATHS)
	@echo '     "raw" : "$(BATHYMETRY_FOLDER)/raw/",' >> $(CONFIG_FILE_PATHS)
	@echo '     "merged" : "$(BATHYMETRY_FOLDER)/merged.npy"' >> $(CONFIG_FILE_PATHS)
	@echo '	},' >> $(CONFIG_FILE_PATHS)
	@echo '  "land" : {' >> $(CONFIG_FILE_PATHS)
	@echo '		"folder" : "$(LAND_FOLDER)",' >> $(CONFIG_FILE_PATHS)
	@echo '		"base" : "$(LAND_FOLDER)/base/",' >> $(CONFIG_FILE_PATHS)
	@echo '     "raw" : "$(LAND_FOLDER)/raw/",' >> $(CONFIG_FILE_PATHS)
	@echo '     "zip" : "$(LAND_FOLDER)/zip/",' >> $(CONFIG_FILE_PATHS)
	@echo '     "coord_boundaries" : "$(LAND_FOLDER)/coord_boundaries.csv"' >> $(CONFIG_FILE_PATHS)
	@echo '	},' >> $(CONFIG_FILE_PATHS)
	@echo '  "interpolated" : {' >> $(CONFIG_FILE_PATHS)
	@echo '		"folder" : "$(INTERPOLATED_FOLDER)",' >> $(CONFIG_FILE_PATHS)
	@echo '		"base" : "$(INTERPOLATED_FOLDER)/base/",' >> $(CONFIG_FILE_PATHS)
	@echo '     "zip" : "$(INTERPOLATED_FOLDER)/zip/"' >> $(CONFIG_FILE_PATHS)
	@echo '	}' >> $(CONFIG_FILE_PATHS)
	@echo '}' >> $(CONFIG_FILE_PATHS)

preprocess : config
	@echo "\nCreating train dataset with $(DATASET_CREATION_CODE_PATH) code..."
	$(PYTHON) $(DATASET_CREATION_CODE_PATH) --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)

regrid:
	@echo "Regridding the dataset..."
	$(PYTHON) $(PREPROCESSING_FOLDER)/regrid.py --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)

train: config
	@echo "Training the model..."
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
