include ../../Makefile

SCRIPTS_FOLDER := $(SRC_DIR)/preprocessing/bathymetry

DATA_FOLDER := $(BATHYMETRY_DATA_DIR)
DOWNLOADED_FILES_FOLDER := $(DATA_FOLDER)/raw
ZIP_FOLDER := $(DATA_FOLDER)/zip
PROCESSED_FILES_FOLDER := $(DATA_FOLDER)/base
NETCDF_FOLDER := $(DATA_FOLDER)/netcdf
NPY_FOLDER := $(DATA_FOLDER)/npy
MERGED_IMAGE_PATH := $(DATA_FOLDER)/merged.npy

PLOT_FOLDER := $(BASE_DIR)/figs/bathymetry

.PHONY: downoad run plot zip unzip help

download:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS_FOLDER)/download.py $(DOWNLOADED_FILES_FOLDER)/

run: config
	@if [ ! -d "$(NETCDF_FOLDER)" ] || [ -z "$(wildcard $(NETCDF_FOLDER)/*)" ]; then \
		make unzip_dowloaded; \
	fi
	@if [ ! -d "$(NPY_FOLDER)" ] || [ -z "$(wildcard $(NPY_FOLDER)/*)" ]; then \
		make netcdf_to_numpy; \
	fi
	@ if [ ! -f "$(MERGED_IMAGE_PATH)" ]; then \
		make merge; \
	fi
	make cut


unzip_dowloaded:
	@echo "Unzipping downloaded data\n"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(UNZIP_CODE) $(DOWNLOADED_FILES_FOLDER)/ $(NETCDF_FOLDER)/

netcdf_to_numpy:
	@echo "Converting netcdf files to numpy arrays"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS_FOLDER)/netcdf_to_numpy.py $(NETCDF_FOLDER)/ $(NPY_FOLDER)/

merge:
	@ echo "Merging numpy arrays\n"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS_FOLDER)/merge.py $(NPY_FOLDER)/ $(MERGED_IMAGE_PATH)

cut:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON)  $(SCRIPTS_FOLDER)/cut_along_coastlines.py $(MERGED_IMAGE_PATH) $(PROCESSED_FILES_FOLDER)/ $(CONFIG_FILE)

plot:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS_FOLDER)/plot.py $(PROCESSED_FILES_FOLDER)/ $(PLOT_FOLDER)/

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
	@echo "  download: Download the bathymetry data from EMODnet"
	@echo "  run: Convert the downloaded data to the format used in the project"
	@echo "  zip: zip the converted data in groups of $(GROUP_SIZE)"
	@echo "  unzip: Unzip the zipped data"
	@echo "  help: Show this help message"
