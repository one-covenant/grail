#!/bin/bash

# Script to compile GRAIL Technical Report
# Requires LaTeX installation

echo "GRAIL Technical Report Compilation Script"
echo "========================================="
echo ""

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex is not installed."
    echo ""
    echo "To install LaTeX on Ubuntu/Debian:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended"
    echo ""
    echo "To install LaTeX on macOS:"
    echo "  brew install --cask mactex"
    echo ""
    echo "Or install BasicTeX (smaller):"
    echo "  brew install --cask basictex"
    echo ""
    exit 1
fi

# Change to docs directory
cd "$(dirname "$0")"

echo "Compiling grail_technical_report.tex..."
echo ""

# First pass
pdflatex grail_technical_report.tex

# Run BibTeX if .bib file exists
if [ -f references.bib ]; then
    echo ""
    echo "Processing bibliography..."
    bibtex grail_technical_report
fi

# Second pass (resolve references)
pdflatex grail_technical_report.tex

# Third pass (finalize)
pdflatex grail_technical_report.tex

# Clean up auxiliary files
echo ""
echo "Cleaning up auxiliary files..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc

echo ""
echo "Done! Output: grail_technical_report.pdf"
echo ""
echo "Authors updated to:"
echo "  - distributed_tensor"
echo "  - const"
echo "  - Yuma Rao"