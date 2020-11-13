# expreriment 1
# CUDA_VISIBLE_DEVICES=0 python train.py --learn-rate 0.001 --weight-decay 0 --batch-size 4 --gpu-ids 0 --backbone unet_siis --dataset /data/home/zy/huawei

CUDA_VISIBLE_DEVICES=0 python train.py --learn-rate 0.001 --weight-decay 0 --gpu-ids 0  --backbone unet --dataset ./data/ \
--mixed_precision no_use  --resume ./ckpt/unet/11-13-11_02_27/model_best.pth \
--base-size 256 --crop-size 256 --epochs 100 --batch-size 2
