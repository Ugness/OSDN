#!/usr/bin/env bash
for i in $(ls ../OSR/osdn/features/ | grep baseline);
do
    echo "../OSR/osdn/features/${i}"
    python eval_openmax.py --synsetfname ../OSR/osdn/cifar6_synsets.txt --image_folder "../OSR/osdn/features/${i}"
done