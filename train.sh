# expreriment 1
# CUDA_VISIBLE_DEVICES=0 python train.py --learn-rate 0.001 --weight-decay 0 --batch-size 4 --gpu-ids 0 --backbone unet_siis --dataset /data/home/zy/huawei

CUDA_VISIBLE_DEVICES=0,1 python train.py --learn-rate 0.001 --weight-decay 0 --batch-size 4 --gpu-ids 0,1 --backbone unet_siis --dataset /data/home/zy/huawei /
--base-size 256 --crop-size 256 --epochs 100 --batch-size 32
