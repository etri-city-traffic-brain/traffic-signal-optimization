#!/bin/sh
DEFAULT_SCENARIO_FILE=/uniq/simulator/salt/sample/sample.json
SCENARIO_FILE=${1:-$DEFAULT_SCENARIO_FILE}
python /uniq/simulator/salt/bin/salt.py -s $SCENARIO_FILE
