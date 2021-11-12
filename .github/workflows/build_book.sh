#!/usr/bin/env bash
set -xe

jupyter_book_dir=book
jupyter_book_build_dir="$jupyter_book_dir/_build/html"

function show_error_logs {
    echo "Some notebooks failed, see logs below:"
    for f in $jupyter_book_build_dir/reports/*.log; do
        echo "================================================================================"
        echo $f
        echo "================================================================================"
        cat $f
    done
    # You need to exit with non-zero here to cause the build to fail
    exit 1
}

jupyter-book build $jupyter_book_dir

# Grep the log to make sure there has been no errors when running the notebooks
# since jupyter-book exit code is always 0
grep 'Execution Failed' $jupyter_book_dir/build.log && show_error_logs || \
    echo 'All notebooks ran successfully'
