# Research Analysis Makefile
#
# Usage:
#   make figures    - Regenerate all figures
#   make clean      - Remove generated figures
#   make help       - Show this help message

.PHONY: figures clean help

PYTHON := python
SCRIPTS_DIR := scripts
FIGURES_DIR := ../figures

help:
	@echo "Research Analysis Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make figures    - Regenerate all figures"
	@echo "  make clean      - Remove generated figures"
	@echo "  make learning   - Generate learning curve figures only"
	@echo "  make benchmarks - Generate benchmark figures only"
	@echo "  make comparisons- Generate comparison figures only"

figures:
	@echo "Generating all figures..."
	cd $(CURDIR) && $(PYTHON) -m scripts.generate_all

learning:
	@echo "Generating learning curve figures..."
	cd $(CURDIR) && $(PYTHON) -m scripts.learning_curves

benchmarks:
	@echo "Generating benchmark figures..."
	cd $(CURDIR) && $(PYTHON) -m scripts.benchmark_bars

comparisons:
	@echo "Generating comparison figures..."
	cd $(CURDIR) && $(PYTHON) -m scripts.comparisons

clean:
	@echo "Removing generated figures..."
	rm -f $(FIGURES_DIR)/*.png
	@echo "Done."

