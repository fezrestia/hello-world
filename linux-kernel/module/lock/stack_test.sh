#!/bin/bash

TOPDIR=/sys/kernel/debug/teststack

for ((i=0; i<1000; ++i)) ; do
    echo 0 > ${TOPDIR}/push &
    cat ${TOPDIR}/show > /dev/null &
    cat ${TOPDIR}/pop > /dev/null &
done

