MAKE = make

SRC_FOLDER = ../
PREPROCESSING_FOLDER = $(SRC_FOLDER)/data_preprocessing
DATA_FOLDER = ../data
DATASET_FOLDER = $(DATA_FOLDER)/datasets
TINY_IMAGENET_FOLDER = ../../tiny-imagenet-200/train
N_TRAIN = 10000
N_TEST = 1000
N_CLASSES = 50
HOLE_PERCENTAGE = 5

PYTHONPATH = $(SRC_FOLDER)

.PHONY: download run clean zip unzip help
.DEFAULT:
	@echo "Invalid target. Please run '$(MAKE) help' for more information."

run:
	@echo "Creating train dataset..."
	python3 $(PREPROCESSING_FOLDER)/create_dataset.py $(TINY_IMAGENET_FOLDER) $(DATASET_FOLDER)/train $(DATASET_FOLDER)/test $(N_TRAIN) $(N_TEST) $(N_CLASSES) $(HOLE_PERCENTAGE)

zip:
	@echo "Zipping dataset..."
	PYTHONPATH=$(PYTHONPATH) python3 $(PREPROCESSING_FOLDER)/zip_datasets.py $(DATASET_FOLDER)/train $(DATASET_FOLDER)/test

unzip:
	@echo "Unzipping dataset..."
	PYTHONPATH=$(PYTHONPATH) python3 $(PREPROCESSING_FOLDER)/unzip_datasets.py $(DATASET_FOLDER)/train $(DATASET_FOLDER)/test

clean:
	@echo "Cleaning non zipped files..."
	rm -rf $(DATASET_FOLDER)/train/*.pth
	rm -rf $(DATASET_FOLDER)/test/*.pth

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  run: Create train dataset from tiny-imagenet-200"
	@echo "  zip: Zip train and test datasets"
	@echo "  unzip: Unzip train and test datasets"
	@echo "  clean: Clean non zipped datasets"
	@echo "  help: Show this help message"
