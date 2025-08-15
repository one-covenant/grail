# Makefile for GRAIL Technical Report

LATEX = pdflatex
BIBTEX = bibtex
RM = rm -f

MAIN = grail_technical_report
SOURCES = $(MAIN).tex
BIBFILE = references.bib

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES) $(BIBFILE)
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	$(RM) *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	$(RM) $(MAIN).pdf

.PHONY: all clean distclean