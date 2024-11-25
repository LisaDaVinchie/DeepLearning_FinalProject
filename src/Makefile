SRC_FOLDER = $(shell pwd)
PYTHONPATH = $(SRC_FOLDER)

DATASET_FOLDER = ../data/datasets
TRAIN_PATH = $(DATASET_FOLDER)/train/dataset_2000_100_5.pth
TEST_PATH = $(DATASET_FOLDER)/test/dataset_500_100_5.pth

MODELS_FOLDER = $(SRC_FOLDER)/models


.PHONY: run help zip unzip clean
.DEFAULT:
	@echo "Invalid target. Please run 'make help' for more information."

run:
	@echo "Training the model..."
	PYTHONPATH=$(PYTHONPATH) python3 $(MODELS_FOLDER)/train.py $(TRAIN_PATH) $(TEST_PATH)

zip:
	@echo calling make -f preprocessing.mk zip
	make -f preprocessing.mk zip

unzip:
	@echo calling make -f preprocessing.mk unzip
	make -f preprocessing.mk unzip

clean:
	@echo calling make -f preprocessing.mk clean
	make -f preprocessing.mk clean

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  run: Train the model"
	@echo "  help: Show this help message"