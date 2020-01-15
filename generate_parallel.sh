#!/bin/bash - 
#===============================================================================
#
#          FILE: generate_parallel.sh
# 
#         USAGE: ./generate_parallel.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 2019-12-17 17:48
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

END=1
for ((i=1;i<=END;i++)); do
    echo $i;
    CUDA_VISIBLE_DEVICES="0" python generator_network.py;
done &
sleep 5
for ((i=1;i<=END;i++)); do
    echo $i;
    CUDA_VISIBLE_DEVICES="1" python generator_network.py;
done &
sleep 5
for ((i=1;i<=END;i++)); do
    echo $i;
    CUDA_VISIBLE_DEVICES="2" python generator_network.py;
done &
sleep 5
for ((i=1;i<=END;i++)); do
    echo $i;
    CUDA_VISIBLE_DEVICES="0" python generator_network.py;
done &
sleep 5
for ((i=1;i<=END;i++)); do
    echo $i;
    CUDA_VISIBLE_DEVICES="1" python generator_network.py;
done &
sleep 5
for ((i=1;i<=END;i++)); do
    echo $i;
    CUDA_VISIBLE_DEVICES="2" python generator_network.py;
done &
sleep 5
