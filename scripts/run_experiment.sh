#!/bin/bash
TIME=6
PYTHONSCRIPT=main.py
END=true
ENV=true
re='^[0-9]+$'

for var in "$@"
do
    if [[ $var == no-end ]]; then
        END=false
    elif [[ $var == no-env ]]; then
        ENV=false
    elif [[ $var =~ $re ]]; then
        TIME=$var
    elif [[ $var == *.py ]]; then
        PYTHONSCRIPT=$var
    fi
done

echo "Running with the following setup:"
echo "Time limit: ${TIME}h"
echo "Python script: $PYTHONSCRIPT"
echo "Close server when done: $END"
echo "Run in virtualenv: $ENV"
echo ""

cd ~/AMD_Detection
if [[ $ENV == true ]]; then
    source ~/.env/neural-nets-utokyo/bin/activate
fi
timeout ${TIME}h python3 ${PYTHONSCRIPT}
if [[ $ENV == true ]]; then
    deactivate
fi
if [[ $END == true ]]; then
    ms end
fi