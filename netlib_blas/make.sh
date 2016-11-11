#!/bin/bash

CC=gcc
FC=ftn
FFLAGS="-O3 -march=native"

for file in *.f; do $FC $FFLAGS -c $file; done

# This is to sanity check that we can successfully link all these files.
$CC main.cc dgemm.o  dgesv.o  dgetrf.o  dgetrf2.o  dgetrs.o  dlaswp.o  dtrsm.o  xerbla.o lsame.o ilaenv.o idamax.o dlamch.o dscal.o ieeeck.o iparmq.o -lm -lgfortran -o main
