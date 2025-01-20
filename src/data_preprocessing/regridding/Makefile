include ../../Makefile

SCRIPTS_FOLDER := $(SRC_DIR)/preprocessing/regridding
TEST_FOLDER := $(SRC_DIR)/tests/preprocessing

BATH_DATA_FOLDER := $(BATHYMETRY_DATA_DIR)/base
LAND_DATA_FOLDER := $(LAND_DATA_DIR)/base
LAND_COORD_BOUNDARIES_FILE := $(LAND_DATA_DIR)/coord_boundaries.csv

PROCESSED_FILES_FOLDER := $(INTERPOLATED_DATA_DIR)/base
ZIP_FOLDER := $(INTERPOLATED_DATA_DIR)/zip

.PHONY: run test zip unzip help

run: config
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS_FOLDER)/run.py $(BATH_DATA_FOLDER)/ $(LAND_DATA_FOLDER)/ $(LAND_COORD_BOUNDARIES_FILE) $(PROCESSED_FILES_FOLDER)/ $(CONFIG_FILE)

test:
	PYTHONPATH=$(PYTHONPATH) pytest $(TEST_FOLDER)/test_interpolation.py

zip:
	@if [ -d "$(ZIP_FOLDER)/" ] && [ "$(wildcard $(ZIP_FOLDER)/*)" ]; then \
		rm -r $(ZIP_FOLDER)/*; \
		echo "\nExisting files in $(ZIP_FOLDER) have been removed.\n"; \
	fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(ZIP_CODE) $(PROCESSED_FILES_FOLDER)/ $(ZIP_FOLDER)/ $(ZIP_SIZE_LIMIT_MB)

unzip:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(UNZIP_CODE) $(ZIP_FOLDER)/ $(PROCESSED_FILES_FOLDER)/

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  run: interpolate the land data"
	@echo "  test: run the tests"
	@echo "  zip: zip the interpolated data into a file or group of files"
	@echo "  help: Show this help message"