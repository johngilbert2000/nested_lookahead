#!/bin/bash
j=0
for opt_t in "SGD" "Adam"
do
    for i in {1..2}
    do
        epochs=20
        bs=64
        lr=0.001
        momentum=0.9
        weight_decay=0
        k=5
        a=0.5
        s=5
        h=0.5
        pullback="None"
        echo "($j) $opt_t $i (wd=$weight_decay)"
        echo "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag wd0_${i}"
        eval "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag wd0_${i}"
        j=$((j+1))
    done
    for i in {1..2}
    do
        epochs=20
        bs=64
        lr=0.001
        momentum=0
        weight_decay=0
        k=5
        a=0.5
        s=5
        h=0.5
        pullback="None"
        echo "($j) $opt_t $i (momentum=$momentum, wd=$weight_decay)"
        echo "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag mom0_wd0_${i}"
        eval "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag mom0_wd0_${i}"
        j=$((j+1))
    done
    for i in {1..2}
    do
        epochs=20
        bs=64
        lr=0.0005
        momentum=0.9
        weight_decay=0.001
        k=5
        a=0.5
        s=5
        h=0.5
        pullback="None"
        echo "($j) $opt_t $i (lr=$lr)"
        echo "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag half_lr_${i}"
        eval "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag half_lr_${i}"
        j=$((j+1))
    done
    for i in {1..2}
    do
        epochs=20
        bs=64
        lr=0.005
        momentum=0.9
        weight_decay=0.001
        k=5
        a=0.5
        s=5
        h=0.5
        pullback="None"
        echo "($j) $opt_t $i (lr=$lr)"
        echo "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag 5x_lr_${i}"
        eval "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag 5x_lr_${i}"
        j=$((j+1))
    done
done
for opt_t in "Lookahead_SGD" "Lookahead_Adam" "NestedLookahead_SGD" "NestedLookahead_Adam"
do
    for i in {1..2}
    do
        epochs=20
        bs=64
        lr=0.001
        momentum=0.9
        weight_decay=0.001
        k=5
        a=0.3
        s=5
        h=0.7
        pullback="None"
        echo "($j) $opt_t $i (a=$a, h=$h)"
        echo "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag a03_h07_${i}"
        eval "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag a03_h07_${i}"
        j=$((j+1))
    done
    for i in {1..2}
    do
        epochs=20
        bs=64
        lr=0.001
        momentum=0.9
        weight_decay=0.001
        k=5
        a=0.7
        s=5
        h=0.3
        pullback="None"
        echo "($j) $opt_t $i (a=$a, h=$h)"
        echo "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag a07_h03_${i}"
        eval "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag a07_h03_${i}"
        j=$((j+1))
    done
    for i in {1..2}
    do
        epochs=20
        bs=64
        lr=0.001
        momentum=0.9
        weight_decay=0.001
        k=5
        a=0.5
        s=10
        h=0.5
        pullback="None"
        echo "($j) $opt_t $i (k=$k, s=$s)"
        echo "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag k5_s10_${i}"
        eval "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag k5_s10_${i}"
        j=$((j+1))
    done
    for i in {1..2}
    do
        epochs=20
        bs=64
        lr=0.001
        momentum=0.9
        weight_decay=0.001
        k=10
        a=0.5
        s=5
        h=0.5
        pullback="None"
        echo "($j) $opt_t $i (k=$k, s=$s)"
        echo "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag k10_s5_${i}"
        eval "python experiment_cifar10.py --epochs $epochs --bs $bs --lr $lr --mom $momentum --wd $weight_decay --opt $opt_t --k $k --a $a --s $s --h $h --pullback $pullback --tag k10_s5_${i}"
        j=$((j+1))
    done
done
