#!/bin/bash
for opt_t in "SGD" "Adam" "Lookahead_SGD" "Lookahead_Adam" "NestedLookahead_SGD" "NestedLookahead_Adam"
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
        pullback="None"
        echo "($j) $opt_t $i"
        echo "python run_cifar.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag $i"
        eval "python run_cifar.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag $i"
    done
done