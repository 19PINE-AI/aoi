MAIN := AnonymousSubmission2027
TEX := $(MAIN).tex
PDF := $(MAIN).pdf

LATEXMK ?= latexmk
LATEXMKFLAGS := -pdf -bibtex -interaction=nonstopmode -halt-on-error -file-line-error

BIB_FILES := $(wildcard *.bib)
STYLE_FILES := $(wildcard *.bst *.sty)
FIGURE_FILES := $(wildcard figs/*.pdf figures/*.pdf figures/*.png)

.PHONY: all pdf clean distclean

all: pdf

pdf: $(PDF)

$(PDF): $(TEX) $(BIB_FILES) $(STYLE_FILES) $(FIGURE_FILES)
	$(LATEXMK) $(LATEXMKFLAGS) $(TEX)

clean:
	$(LATEXMK) -c $(TEX)

distclean:
	$(LATEXMK) -C $(TEX)
