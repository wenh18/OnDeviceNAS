#bash distributed_train_resnet.sh 2 \--dataset imagenet \-b 104 \--model resnet50d \--epochs 50 \--amp \--remode pixel \--reprob 0.6 \--aug-splits 3 \--aa rand-m9-mstd0.5-inc1 \--resplit \--split-bn \--jsd \--dist-bn reduce \--num-classes 1000 \--data_dir ../../../../../liyc/data/imagenet2012/ \--opt adam \--decay-epochs 1 \--decay-rate .95 \--sched step \--lr 3e-4 \--warmup-lr 5e-4 \--savemodelfreq 1 \--withcheckpoint \--checkpointdir output/resnet_weights/resnet_3epoch7.pth
# python train_resnet.py \--dataset imagenet \-b 120 \--model resnet50d \--amp \--remode pixel \
# --reprob 0.6 \--aug-splits 3 \--aa rand-m9-mstd0.5-inc1 \--jsd \--num-classes 1000 \
# --data_dir ../../../../../liyc/data/imagenet2012/ \--opt adam \--decay-epochs 1 \--decay-rate .95 \
# --sched step \--lr 3e-4 \--warmup-lr 5e-3 \--warmup-epochs 1 \--savemodelfreq 1 \
# --original_model pretrainedweight/resnet50d_ra2-464e36ba.pth \--distilled_model output/resnet_weights/resnet_0epoch33acc61.pth \
# --stage 0  \--start-epoch 0 \--pretrain_epochs 25  \--start-epoch1 15 \--epochs 40 \--resplit \--split-bn
 # \--dist-bn reduce # \--resplit \--split-bn \--dist-bn reduce  
# \--resplit \--split-bn \--dist-bn reduce  \--withcheckpoint \--distilled_model output/resnet_weights/resnet_3epoch9.pth bash distributed_train_resnet.sh 2
python train_resnet.py \--dataset imagenet --model resnet50 --amp --batch 352 --decay-epochs 2 \
--decay-rate .9 --lr 1.35e-3 --epochs 600 --weight-decay 0.01 --sched step --train-interpolation bicubic \
 --crop-pct 0.95 --smoothing 0.1 --warmup-epochs 2 --aa rand-m7-n3-mstd1.0-inc1 --seed 0 --opt adamp --warmup-lr 1.5e-3 \
  --drop-path 0.05 --drop 0.1 --reprob 0.35 --mixup .2 --cutmix 1.0 --bce-loss --pretrain_epochs 40 --original_model pretrainedweight/resnet50_a1_0-14fe96d1.pth --data_dir ../../../../../liyc/data/imagenet2012/
# origin lr -- 3.3e-3, batch size = 192*2  lr scaling:0.2777777777