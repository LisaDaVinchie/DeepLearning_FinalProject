include ../../Makefile

.PHONY: run help

# Run the model
run: config
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(MODELS_FOLDER)/train.py --paths $(CONFIG_FILE_PATHS) --params $(CONFIG_FILE_PARAMS)

help:
	@echo "Makefile for the Transformer model"
	@echo ""
	@echo "Usage:"
	@echo "  make run          Run the model"
	@echo "  make help         Show this message"
	@echo ""

