all:
	pdflatex gft_proofs.tex
	bibtex gft_proofs
	pdflatex gft_proofs.tex
	pdflatex gft_proofs.tex
	python3 run_validation.py
