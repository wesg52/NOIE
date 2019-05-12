#!/usr/bin/env bash
python benchmark.py --gold=./oie_corpus/all.oie --out=./transformer.dat --stanford=./systems_output/extractions2.txt
mv transformer.dat result
python pr_plot.py --in=results --out=all.png