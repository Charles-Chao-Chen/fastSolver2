#!/bin/bash

for file in *.f; do ftn -O3 -c $file; done
gcc main.cc dgemm.o  dgesv.o  dgetrf.o  dgetrf2.o  dgetrs.o  dlaswp.o  dtrsm.o  xerbla.o lsame.o ilaenv.o idamax.o dlamch.o dscal.o ieeeck.o iparmq.o -lm -lgfortran -o main
