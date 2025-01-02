include ../Makefile


.PHONY: run help

run: config
	@echo "Training the model..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(MODELS_FOLDER)/train.py --n_train $(N_TRAIN) --n_test $(N_TEST) --n_classes $(N_CLASSES) --mask_percentage $(MASK_PERCENTAGE) --config_path $(CONFIG_FILE)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  run: Train the model"
	@echo "  help: Show this help message"