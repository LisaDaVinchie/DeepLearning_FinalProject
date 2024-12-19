SRC_FOLDER = $(shell pwd)
PYTHONPATH = $(SRC_FOLDER)

DATASET_FOLDER = ../data/datasets

N_TRAIN = 10000
N_TEST = 1000
N_CLASSES = 50
HOLE_PERCENTAGE = 5

TRAIN_PATH = $(DATASET_FOLDER)/train/dataset_$(N_TRAIN)_$(N_CLASSES)_$(HOLE_PERCENTAGE).pth
TEST_PATH = $(DATASET_FOLDER)/test/dataset_$(N_TEST)_$(N_CLASSES)_$(HOLE_PERCENTAGE).pth

MODELS_FOLDER = $(SRC_FOLDER)/models


.PHONY: run help zip unzip clean
.DEFAULT:
	@echo "Invalid target. Please run 'make help' for more information."

run:
	@echo "Training the model..."
	PYTHONPATH=$(PYTHONPATH) python3 $(MODELS_FOLDER)/train.py $(TRAIN_PATH) $(TEST_PATH)

zip:
	@echo calling make -f preprocessing.mk zip
	cd data_preprocessing && make zip

unzip:
	@echo calling make -f preprocessing.mk unzip
	cd data_preprocessing && make unzip

clean:
	@echo calling make -f preprocessing.mk clean
	cd data_preprocessing && make clean

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  run: Train the model"
	@echo "  help: Show this help message"