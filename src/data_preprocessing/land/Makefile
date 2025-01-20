include ../../Makefile

DATA_FOLDER = $(LAND_DATA_DIR)
SCRIPTS_FOLDER := $(SRC_DIR)/preprocessing/land

DOWNLOADED_FILES_FOLDER := $(DATA_FOLDER)/raw
PROCESSED_FILES_FOLDER := $(DATA_FOLDER)/base
ZIP_FOLDER := $(DATA_FOLDER)/zip

DOWNLOAD_ID := de5e20065fa0460a8d6281246429b272

.PHONY: downoad run zip unzip help

download:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS_FOLDER)/download.py $(DOWNLOAD_ID) $(DOWNLOADED_FILES_FOLDER)/

run: config
	@echo "the downloaded files folder is $(DOWNLOADED_FILES_FOLDER)\n"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS_FOLDER)/run.py $(DOWNLOADED_FILES_FOLDER)/ $(PROCESSED_FILES_FOLDER)/ $(ZIP_FOLDER)/

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
	@echo "  download: Download the land data from Copernicus"
	@echo "  run: Convert the downloaded data to the format used in the project"
	@echo "  unzip: Unzip the zipped data"
	@echo "  help: Show this help message"



