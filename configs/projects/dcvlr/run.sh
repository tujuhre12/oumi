export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_TOKEN=""
export CUDA_LAUNCH_BLOCKING="1"

oumi distributed torchrun --nproc-per-node=4 -m oumi train -c configs/projects/dcvlr/molmo.yaml