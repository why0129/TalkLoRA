export CUDA_VISIBLE_DEVICES=0


python train_talklora.py \
  --peft_type=talklora \
  --model=model_path \
  --r_ab=32 \
  --enable_grad_ckpt \
  --epoch=2 \
  --lr=3e-4 \
  --batch=8 \
  --grad_acc=2 \
  --dataset=common_170k \
  --seed=36 \
  --warmup=100 \
  --eval_strategy=steps \
  --eval_steps=80 \
  --output_folder=path \
  --target_module=q_proj,k_proj,v_proj,up_proj,down_proj




