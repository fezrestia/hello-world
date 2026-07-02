# Deep Learning from ZERO

## Env Setup

    $ apt install python3-venv
    $ python3 -m venv venv

    # activate venv
    $ source venv/bin/activate

    # deactivate venv
    $ deactivate

## Export PKG Dependency

    $ pip3 freeze > dependencies.txt

## Import PKG Dependency

    # after env setup
    $ pip3 install -r dependencies.txt

## Additional PKG

    $ sudo apt install graphviz
    $ sudo apt install gymnasium

## Test

    $ cd framework/

    # for each test file.
    $ python3 -m unittest deepzero.test.test_basic_math

    # for all test files under test/.
    $ python3 -m unittest discover -s deepzero/test -p "test_*.py" -t . -v

    # or
    $ ./test.sh

