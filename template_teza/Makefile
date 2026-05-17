TEX_FILE = teza.tex
PDF_FILE = teza.pdf
BIB_FILE = referinte

.PHONY: all clean

all: $(PDF_FILE)

$(PDF_FILE): $(TEX_FILE) chapters/*.tex $(BIB_FILE).bib
	pdflatex -synctex=1 -interaction=nonstopmode $(TEX_FILE)
	bibtex $(basename $(TEX_FILE))
	pdflatex -synctex=1 -interaction=nonstopmode $(TEX_FILE)
	pdflatex -synctex=1 -interaction=nonstopmode $(TEX_FILE)

clean:
	rm -f *.out *~ *.thm *.log *.bbl *.blg *.aux *.toc *.lot *.lof *.dvi *.ps *.synctex.gz *.pdf
