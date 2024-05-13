CKPT_NAME="llava-v1.5-7b"
num_frames=4
model_path="ckpt/llava-v1.5-7b"
GPT_Zero_Shot_QA="/bpfs/v2_mnt/VIS/wuwenhao/mllm_data/GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/VideoChatGPT_Test_Videos"
gt_file="${GPT_Zero_Shot_QA}/temporal_qa.json"
output_dir="output/Benchmark_Temporal_QA/${CKPT_NAME}_u${num_frames}FRS"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 llava/eval/run_inference_benchmark_general.py \
      --video_dir ${video_dir} \
      --gt_file ${gt_file} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --model_name ${model_path} \
      --num_chunks $CHUNKS \
      --num_frames $num_frames \
      --conv-mode vicuna_v1 \
      --temperature 0 \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/temporal.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done