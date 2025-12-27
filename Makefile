PAPER = paper
TEX = $(wildcard *.tex)
BIB = refs.bib
SCRIPTS = scripts/generate_plot_data.py
DATA = data/perf_results.csv
PYTHON = .venv/bin/python3

.PHONY: all clean data

all: $(PAPER).pdf

# Generate plot data from CSV before building PDF
data: $(DATA) $(SCRIPTS)
	$(PYTHON) scripts/generate_plot_data.py

artifacts/throughput_comparison.csv: $(DATA) $(SCRIPTS)
	$(PYTHON) scripts/generate_plot_data.py

$(PAPER).pdf: $(TEX) $(BIB) artifacts/throughput_comparison.csv
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf
	rm -rf artifacts/
