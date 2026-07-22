# setup

    $ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

    $ source ~/.cargo/env

    $ rustc --version
    $ cargo --version
    $ rustup --version


# project

    $ cargo new hello-world
    $ cd hello-world


# build

    $ cargo build


# exec

    $ ./target/debug/hello-world

or

    $ cargo run


# release build and exec

    $ cargo build --release

    $ ./target/release/hello-world


# component -> add to rust env

    $ rustup component add rustfmt
    $ rustup component add clippy


# dependency -> add to rust project

    $ cargo add rand


# binding with python

    $ pip install maturin

    $ maturin init --bindings pyo3 hello_world_pyo3
    $ cd hello_world_pyo3

    $ maturin build

