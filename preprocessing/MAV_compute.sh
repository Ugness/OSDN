#!/usr/bin/env bash
for i in $(find ../../OSR/osdn/features/ -wholename '*/train');
do
    for j in $(seq 0 9);
    do
        python MAV_Compute.py  --synset $j -feat $i -list ~/jhyoo/OSR/osdn/cifar10_synsets.txt
    done
done