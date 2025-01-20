include $(SRC_DIR)/Makefile

# Declare where the scripts are
SCRIPTS_FOLDER := $(SRC_DIR)/preprocessing/dataset_masking
TEST_FOLDER := $(SRC_DIR)/tests/preprocessing

# Declare the data directories
BATH_DATA_FOLDER := $(BATHYMETRY_DATA_DIR)/base
MASKED_DATA_FOLDER := $(MASKS_DATA_DIR)/base
INTERP_DATA_FOLDER := $(INTERPOLATED_DATA_DIR)/base
ZIP_FOLDER := $(MASKS_DATA_DIR)/zip

FIGS_FOLDER := $(BASE_DIR)/figs/masked_data

# Declare the dictionary filename
DICTIONARY_FILENAME := $(DATASET_DIR)/masked_data_dict.pth

.PHONY: run save plot test help zip unzip

run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS_FOLDER)/run.py $(BATH_DATA_FOLDER)/ $(INTERP_DATA_FOLDER)/ $(MASKED_DATA_FOLDER)/

save:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS_FOLDER)/save_as_dict.py $(INTERP_DATA_FOLDER)/ $(MASKED_DATA_FOLDER)/ $(DICTIONARY_FILENAME)

plot:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS_FOLDER)/plot.py $(MASKED_DATA_FOLDER)/ $(INTERP_DATA_FOLDER)/ $(FIGS_FOLDER)/

zip:
	@if [ -d "$(ZIP_FOLDER)/" ] && [ "$(wildcard $(ZIP_FOLDER)/*)" ]; then \
		rm -r $(ZIP_FOLDER)/*; \
		echo "\nExisting files in $(ZIP_FOLDER) have been removed.\n"; \
	fi
	# PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(ZIP_CODE) $(MASKED_DATA_FOLDER)/ $(ZIP_FOLDER)/ $(ZIP_SIZE_LIMIT_MB)

unzip:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(UNZIP_CODE) $(ZIP_FOLDER)/ $(MASKED_DATA_FOLDER)/
	
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  run: mask the bathymetry data"
	@echo "  test: run the tests"
	@echo "  save: save the masked data as dictionary of torch tensors"
	@echo "  zip: zip the masked data into a file or group of files"
	@echo "  unzip: unzip the zipped data"
	@echo "  help: show this help message"