#!/usr/bin/sh

mypy .

python -m unittest discover -s deepzero/test -p "test_*.py" -t . -v

