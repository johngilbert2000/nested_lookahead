#!/bin/bash
j=0
for opt_t in "Lookahead_SGD" "Lookahead_Adam" "NestedLookahead_SGD" "NestedLookahead_Adam"
do
    for i in {1..3}
    do
        epochs=20
        bs=64
        lr=0.001
        momentum=0.9
        weight_decay=0.001
        k=5
        a=0.5
        s=5
        h=0.5
        for pullback in "reset" "pullback"
        do
            echo "($j) $opt_t $i"
            echo "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag ${pullback}_${i}"
            eval "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag ${$pullback}_${i}"
            j=$((j+1))
        done
    done
done