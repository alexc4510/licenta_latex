TEX_FILE=$(shell ls *.tex)
PDF_FILE=$(TEX_FILE:.tex=.pdf)

.PHONY: all clean
all: $(PDF_FILE)

$(PDF_FILE): $(TEX_FILE)
	pdflatex -synctex=1 -interaction=nonstopmode $<

clean: 
	rm -f *.out *~ *.thm *.log *.bbl *.blg *.aux *.toc *.lot *.lof *.dvi *.ps *.synctex.gz *.pdf


