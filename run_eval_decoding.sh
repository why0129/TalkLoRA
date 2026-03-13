CUDA_VISIBLE_DEVICES=2 python train_talklora.py --dataset=boolq --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=2 python train_talklora.py --dataset=piqa --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=2 python train_talklora.py --dataset=siqa --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=2 python train_talklora.py --dataset=winog --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=2 python train_talklora.py --dataset=arce --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=2 python train_talklora.py --dataset=arcc --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=2 python train_talklora.py --dataset=obqa --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=2 python train_talklora.py --dataset=hellas --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4

