#!/usr/bin/env bash
for i in $(find ../../OSR/osdn/features/ -wholename '*/train');
do
    for j in $(seq 0 9);
    do
        python compute_distances.py  --synset $j -mav "${i}/means/${j}.mat" -feature $i -list ~/jhyoo/OSR/osdn/cifar10_synsets.txt
    done
done