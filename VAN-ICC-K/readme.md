# test_undersampled
CUDA_VISIBLE_DEVICES=1 python 12chto4ch_kspace_test_compress_undersampled.py --task=test --out_path="./exps/" --ckpt="./exps/12chto4ch_cross_smooth_l1_forward_undersampled_cart_R4/checkpoint/latest.pth"
# test_fullsampled
CUDA_VISIBLE_DEVICES=1 python 12chto4ch_kspace_test_compress_fullsampled.py --task=test --out_path="./exps/" --ckpt="./exps/12chto4ch_cross_smooth_l1_forward_fullsampled/checkpoint/latest.pth"
