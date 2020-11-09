# expreriment 1
CUDA_VISIBLE_DEVICES=0 python train.py --learn-rate 0.001 --weight-decay 0 --batch-size 4 --gpu-ids 0 --backbone unet_siis --dataset ./data/