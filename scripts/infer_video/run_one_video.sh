CKPT_NAME="llava-v1.5-13b"
num_frames=8
model_path="ckpt/llava-v1.5-13b"
video_path="video_samples/sample_demo_23.mp4"

CUDA_VISIBLE_DEVICES=0 python3 llava/eval/single_video_inference.py \
      --video_path ${video_path} \
      --model_name ${model_path} \
      --num_frames $num_frames \
      --conv-mode vicuna_v1
      
