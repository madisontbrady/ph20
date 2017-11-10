# Makefile for Ph 20 Assignment 3.
MAKE_TERMOUT = command.txt

include config.mk

.PHONY: all
all : ph20_3.pdf

# Compile pdf.
%.pdf : images log.txt command.txt %.tex
		pdflatex $*.tex -shell-escape
		pdflatex $*.tex -shell-escape

# Produce log.
log.txt :
		git log > $@

# Generate plots.
.PHONY : images
images : $(PLOT_SRC) xv_exp_0_0_20.png xv_exp_1_0_20.png xv_exp_0_1_20.png \
xv_imp_0_1_20.png xv_sym_0_1_20.png er_exp_0_1_100.png er_imp_0_1_100.png \
er_sym_0_1_20.png he_exp_0_1_20.png en_exp_0_1_20.png en_imp_0_1_20.png \
en_sym_0_1_20.png ph_exp_0_1_20.png ph_imp_0_1_20.png ph_sym_0_1_20.png \
lp_sym_0_1_1000.png

%0_0_20.png : $(PLOT_SRC)
		$(PLOT_EXE) $*0_0_20 0.1 0 0 20

%1_0_20.png : $(PLOT_SRC)
		$(PLOT_EXE) $*1_0_20 0.1 1 0 20

%0_1_20.png : $(PLOT_SRC)
		$(PLOT_EXE) $*0_1_20 0.1 0 1 20

%0_1_100.png : $(PLOT_SRC)
		$(PLOT_EXE) $*0_1_100 0.1 0 1 100

%0_1_1000.png : $(PLOT_SRC)
		$(PLOT_EXE) $*0_1_1000 0.1 0 1 1000

.PHONY : clean
clean :
		rm -f *.png
		rm -f *.pdf
		rm -f *.aux
		rm -f *.log
		rm -f *.txt
