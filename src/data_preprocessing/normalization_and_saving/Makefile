include ../../Makefile

SCRIPTS_FOLDER = $(SRC_DIR)/preprocessing/normalization_and_saving

.PHONY: run help

run: config
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS_FOLDER)/run.py $(INTERPOLATED_DATA_DIR)/base $(MASKS_DATA_DIR)/base $(DATASET_DIR)/norm_dict.pth

help:
	@echo "The following are the make targets available in this Makefile:"
	@echo "run: save the normalized data"
	@echo "help"