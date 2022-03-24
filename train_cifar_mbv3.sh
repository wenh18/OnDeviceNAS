python train.py \--dataset imagenet \--model mobilenetv3_large_100 \-b 128 \--epochs 1000 \--decay-epochs 4 \--decay-rate .973 \--opt adam \--opt-eps .001 \-j 7 \--warmup-lr 5e-4 \--weight-decay 1e-5 \--drop 0.2 \--drop-connect 0.2 \--model-ema \--model-ema-decay 0.9999 \--aa rand-m9-mstd0.5 \--remode pixel \--reprob 0.2 \--amp \--lr 3e-4 \--lr-noise 0.042 0.09 \--lr-noise-pct 0.067 \--lr-noise-std 0.1 \--data_dir ../../../../../liyc/data/imagenet2012/ \--pretrain_epochs 0 \--savemodelfreq 5 \--sched step \--min-lr 1e-6 \--num-classes 1000

# bash distributed_train.sh 2 
# terminal on the right:do not have teacher, weight decay is 5e-4, terminal on the left is this bash file, decay is 1e-3  , terminal on the right has been shut down
