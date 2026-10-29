PYTHON ?= python3
TYPST ?= typst
export MPLBACKEND := Agg

.PHONY: all plots pdfs bce-plots
all: plots pdfs

plots:
	$(PYTHON) plot_sigmoid_deriv.py
	$(PYTHON) train.py
	$(PYTHON) train_ln.py
	$(PYTHON) plot_boundary.py

pdfs:
	$(TYPST) compile --input answers=false question.typst backprop.pdf
	$(TYPST) compile question.typst backprop_soln.pdf

bce-plots:
	$(PYTHON) train_bce.py
	$(PYTHON) train_ln_bce.py
