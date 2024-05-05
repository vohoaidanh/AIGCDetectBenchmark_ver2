## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
##export CUDA_VISIBLE_DEVICES=0
##export NCCL_P2P_DISABLE=1
##MODEL_PATH="./weights/preprocessing/lsun_bedroom.pt" # "models/lsun_bedroom.pt, models/256x256_diffusion_uncond.pt"

##SAMPLE_FLAGS="--batch_size 16 --num_samples 2000 --timestep_respacing ddim20 --use_ddim True"
##SAVE_FLAGS="--images_dir /hotdata/share/AIGCDetect/test/DALLE2 --recons_dir ./result/recons/DALLE2 --dire_dir ./result/dire/DALLE2"
##MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
##mpiexec -n 1 python preprocessing/DIRE/compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder True --has_subclasses False

## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
MODEL_PATH="models/64x64_diffusion.pt " # "models/lsun_bedroom.pt, models/256x256_diffusion_uncond.pt"

SAMPLE_FLAGS="--batch_size 16 --num_samples 2000 --timestep_respacing ddim20 --use_ddim True --num_workers 0 "
SAVE_FLAGS="--images_dir dataset/RealFakeDB512s/train --recons_dir ./result/recons/RealFakeDB512s/train --dire_dir ./result/dire/RealFakeDB512s/train "
#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True "

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True "
#python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt --classifier_depth 4 --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS

#mpiexec -n 1 python preprocessing/DIRE/compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder True --has_subclasses False
python preprocessing/DIRE/compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS $SAVE_FLAGS $SAMPLE_FLAGS
# --has_subfolder True --has_subclasses False

