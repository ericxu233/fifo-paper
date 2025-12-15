PAPER = paper
TEX = $(wildcard *.tex)
BIB = refs.bib

.PHONY: all clean

$(PAPER).pdf: $(TEX) $(BIB)
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf
