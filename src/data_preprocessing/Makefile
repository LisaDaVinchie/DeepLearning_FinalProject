include ../Makefile

.PHONY: download zip unzip help

zip:
	@echo "Zipping dataset..."
	PYTHONPATH=$(PYTHONPATH) python3 $(PREPROCESSING_FOLDER)/zip_datasets.py $(DATASET_FOLDER)/train $(DATASET_FOLDER)/test

unzip:
	@echo "Unzipping dataset..."
	PYTHONPATH=$(PYTHONPATH) python3 $(PREPROCESSING_FOLDER)/unzip_datasets.py $(DATASET_FOLDER)/train $(DATASET_FOLDER)/test

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  run: Create train dataset from tiny-imagenet-200"
	@echo "  zip: Zip train and test datasets"
	@echo "  unzip: Unzip train and test datasets"
	@echo "  help: Show this help message"
