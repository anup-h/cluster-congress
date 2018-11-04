#!/bin/bash
cd tweet-preprocessor-0.4.0
python setup.py build
python setup.py install
cd ..
gunicorn app:server
