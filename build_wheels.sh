#!/usr/bin/env fish

rm dist/*.whl

rm -r build
set -xg PYSOUNDCARD_PLATFORM darwin
python setup.py bdist_wheel upload

rm -r build
set -xg PYSOUNDCARD_PLATFORM win32
set -xg PYSOUNDCARD_ARCHITECTURE 32bit
python setup.py bdist_wheel upload

rm -r build
set -xg PYSOUNDCARD_PLATFORM win32
set -xg PYSOUNDCARD_ARCHITECTURE 64bit
python setup.py bdist_wheel upload

rm -r build
set -xg PYSOUNDCARD_PLATFORM noplatform
set -xg PYSOUNDCARD_ARCHITECTURE noarch
python setup.py bdist_wheel upload
python setup.py sdist upload
