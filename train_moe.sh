batch_size=128
val_batch_size=128
num_epochs=500
mse_mult=1000000
model_name="moe_test"
gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
--model_name $model_name --subj_list 1 2 5 7 \
--num_epochs $num_epochs --batch_size $batch_size --val_batch_size $val_batch_size \
--pool_type max --pool_num 8192 \
--mse_mult $mse_mult \
--eval_interval 25 --ckpt_interval 25 \
--max_lr 8e-5 --num_workers 4
