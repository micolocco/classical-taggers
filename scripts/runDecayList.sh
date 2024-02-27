#!/bin/bash
for input in $(cat decays.txt); 
do
    python adding_features.py $input &
    echo '--------------------------------------------------------------------------'
done