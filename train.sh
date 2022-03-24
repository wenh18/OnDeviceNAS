python train.py \--dataset imagenet \--model mobilenetv3_large_100 \-b 256 \--epochs 600 \--decay-epochs 2 \--decay-rate .973 \--opt-eps .001 \-j 7 \--warmup-lr 1e-3 \--weight-decay 1e-5 \--drop 0.2 \--drop-connect 0.2 \--model-ema \--model-ema-decay 0.9999 \--aa rand-m9-mstd0.5 \--remode pixel \--reprob 0.2 \--amp \--lr 3e-4 \--lr-noise 0.042 0.09 \--lr-noise-pct 0.067 \--lr-noise-std 0.01 \--data_dir ../../project1/MultiBranchNet/mobilenet-yolov4-pytorch/pytorch-image-models-master/data/imagenet/ \--pretrain_epochs 0 \--savemodelfreq 5 \--sched step \--min-lr 1e-7 \--num-classes 1000 \--opt adam

# bash distributed_train.sh 1 



