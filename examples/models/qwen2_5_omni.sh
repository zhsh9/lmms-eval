# Run and exactly reproduce qwen2.5-omni results!
# mme as an example
export HF_HOME="~/.cache/huggingface"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip3 install qwen_vl_utils
# use `interleave_visuals=True` to control the visual token position, currently only for mmmu_val and mmmu_pro (and potentially for other interleaved image-text tasks), please do not use it unless you are sure about the operation details.

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_vl \
#     --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=True \
#     --tasks mmmu_pro \
#     --batch_size 1

PRETRAINED=Qwen/Qwen2.5-Omni-7B
BATCH_SIZE=1
OUTPUT_DIR=./results/qwen2_5_omni

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_omni \
    --model_args=pretrained=$PRETRAINED,max_pixels=12845056,attn_implementation=flash_attention_2 \
    --tasks videomme \
    --log_samples --output_path $OUTPUT_DIR/videomme.jsonl \
    --batch_size $BATCH_SIZE

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_omni \
    --model_args=pretrained=$PRETRAINED,max_pixels=12845056,attn_implementation=flash_attention_2 \
    --tasks mlvu \
    --log_samples --output_path $OUTPUT_DIR/mlvu.jsonl \
    --batch_size $BATCH_SIZE

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_omni \
    --model_args=pretrained=$PRETRAINED,max_pixels=12845056,attn_implementation=flash_attention_2 \
    --tasks mvbench \
    --log_samples --output_path $OUTPUT_DIR/mvbench.jsonl \
    --batch_size $BATCH_SIZE
