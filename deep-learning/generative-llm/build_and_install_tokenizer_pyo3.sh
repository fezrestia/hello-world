#!/bin/bash -eux
set -o pipefail

echo "BASH_VERSION = ${BASH_VERSION}"

pushd ./tokenizer_pyo3

    maturin build

    pip install --force-reinstall \
            ./target/wheels/tokenizer_pyo3-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl

popd

