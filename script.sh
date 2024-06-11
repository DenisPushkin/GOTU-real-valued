for seed in {0..4}
do
    # figure 1
    python3 main.py -task const_fn -distr gauss -arch rf --small-features -epsilon 0.05 -num-features 256 -activation poly -deg 2 -dimension 2 -epochs 100 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 1 -verbose-int 1 -training-method GD_line_search
    
    # table 1
    python3 main.py -task 2prod -distr gauss -arch rf -activation shifted_relu -dimension 15 -num-features 40000 -epochs 4000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 65536 -compute-int 20 -verbose-int 20 -training-method GD_line_search
    python3 main.py -task 2prod -distr gauss -arch rf -activation relu -dimension 15 -num-features 40000 -epochs 4000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 65536 -compute-int 20 -verbose-int 20 -training-method GD_line_search
    python3 main.py -task 2prod -distr gauss -arch rf -activation softplus -dimension 15 -num-features 40000 -epochs 5000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 65536 -compute-int 20 -verbose-int 20 -training-method GD_line_search
    python3 main.py -task 2prod -distr gauss -arch rf -activation sigmoid -dimension 15 -num-features 40000 -epochs 2000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 65536 -compute-int 10 -verbose-int 10 -training-method GD_line_search
    
    # table 2
    python3 main.py -task const_fn -distr gauss -arch rf -activation poly --small-features -epsilon 0.03 -deg 2 -dimension 15 -num-features 1024 -epochs 100 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 1 -verbose-int 1 -training-method GD_line_search
    python3 main.py -task const_fn -distr gauss -arch rf -activation shifted_relu --small-features -epsilon 0.03 -dimension 15 -num-features 1024 -epochs 50 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 1 -verbose-int 1 -training-method GD_line_search
    python3 main.py -task const_fn -distr gauss -arch rf -activation softplus --small-features -epsilon 0.03 -dimension 15 -num-features 1024 -epochs 100 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 1 -verbose-int 1 -training-method GD_line_search
    python3 main.py -task const_fn -distr gauss -arch rf -activation sigmoid --small-features -epsilon 0.03 -dimension 15 -num-features 1024 -epochs 100 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 1 -verbose-int 1 -training-method GD_line_search
    python3 main.py -task const_fn -distr gauss -arch rf -activation relu --small-features -epsilon 0.03 -dimension 15 -num-features 1024 -epochs 1000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 5 -verbose-int 5 -training-method GD_line_search
    
    # table 3
    python3 main.py -task const_fn -distr gauss -arch rf -activation poly -deg 2 -dimension 15 -num-features 1024 -epochs 2000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 10 -verbose-int 10 -training-method GD_line_search
    python3 main.py -task const_fn -distr gauss -arch rf -activation shifted_relu -dimension 15 -num-features 1024 -epochs 2000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 10 -verbose-int 10 -training-method GD_line_search
    python3 main.py -task const_fn -distr gauss -arch rf -activation sigmoid -dimension 15 -num-features 1024 -epochs 1000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 10 -verbose-int 10 -training-method GD_line_search
    python3 main.py -task const_fn -distr gauss -arch rf -activation relu -dimension 15 -num-features 1024 -epochs 2000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 10 -verbose-int 10 -training-method GD_line_search
    
    # figure 4
    python3 main.py -task const_fn -distr unif_discrete -support-size 5 -arch rf -activation poly -deg 2 -dimension 15 -num-features 1024 -epochs 2000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 65536 -compute-int 10 -verbose-int 10 -training-method GD_line_search
    
    # figure 5
    python3 main.py -task const_fn -distr unif_discrete -support-size 5 -arch transformer -opt adamw -lr 0.0001 -dimension 15 -epochs 20 -batch-size 256 -batches-per-epoch 100 -seed $seed -valid-size 65536 -test-size 65536 -compute-int 1 -verbose-int 1    
done

for seed in {0..2}
do
    # figure 2
    python3 main.py -task 2geometric -distr gauss -arch rf -activation poly -deg 2 --small-features -epsilon 0.05 -dimension 2 -num-features 16384 -epochs 80000 -batch-size 4096 -seed $seed -train-size 32768 -valid-size 16384 -test-size 32768 -compute-int 200 -verbose-int 200 -training-method GD_line_search
    
    # table 3
    python3 main.py -task const_fn -distr gauss -arch rf -activation softplus -dimension 15 -num-features 1024 -epochs 20000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 50 -verbose-int 50 -training-method GD_line_search
    
    # figure 3
    python3 main.py -task const_fn -distr gauss -arch rf -activation poly -deg 4 -dimension 15 -num-features 300000 -epochs 15000 -batch-size 4096 -seed $seed -train-size 65536 -valid-size 32768 -test-size 32768 -compute-int 50 -verbose-int 50 -training-method GD_line_search

    # figure 6 (left)
    python3 main.py -task 2prod -distr unif_discrete -support-size 3 -arch transformer -opt adamw -lr 0.0001 -dimension 15 -epochs 20000 -batch-size 256 -batches-per-epoch 100 -seed $seed -valid-size 65536 -test-size 131072 -compute-int 50 -verbose-int 50
done

# figure 6 (right)
python3 main.py -task 2prod -distr unif_discrete -support-size 3 -arch transformer -opt adamw -lr 0.00001 -dimension 15 -epochs 60000 -batch-size 256 -batches-per-epoch 100 -seed 0 -valid-size 65536 -test-size 131072 -compute-int 100 -verbose-int 100
