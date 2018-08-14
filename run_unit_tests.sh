#!/bin/bash

topdir=`pwd`

for f in $(find $(pwd) -name test_*.py); do
    cd $( dirname $f)
    echo "Running $f"
    python3  $f
done

cd $topdir
exit
