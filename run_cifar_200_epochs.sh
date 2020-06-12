#!/bin/bash
j=0
for opt_t in "NestedLookahead_Adam" "Lookahead_Adam"
do
    for i in {1..3}
    do
        epochs=200
        bs=64
        lr=0.001
        momentum=0
        weight_decay=0.001
        k=5
        a=0.7
        s=10
        h=0.3
        pullback="None"
        echo "($j) $opt_t $epochs epochs $i"
        echo "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag 200_epochs_${i}"
        eval "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag 200_epochs_${i}"
        j=$((j+1))
    done
done