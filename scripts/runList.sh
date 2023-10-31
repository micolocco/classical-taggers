#!/bin/bash
for input in $(cat taggers.txt); 
do
    python pipeline.py Bu2JpsiK $input &
    echo '--------------------------------------------------------------------------'
done
