CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_mixed.py --learn-rate 0.001 --weight-decay 0 \
--batch-size 16 --gpu-ids 0,1 --backbone unet --dataset /data/home/zy/huawei \
--mixed_precision use \
--base-size 256 --crop-size 256 --epochs 100