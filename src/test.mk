SRC_FOLDER = $(shell pwd)
TEST_FOLDER = $(SRC_FOLDER)/tests
PYTHONPATH = $(SRC_FOLDER)


.PHONY: run

run:
	@echo "Running test..."
	PYTHONPATH=$(PYTHONPATH) pytest $(TEST_FOLDER)/test_autoencoder.py