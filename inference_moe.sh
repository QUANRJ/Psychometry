subj_load=1
subj_test=1
model_name="moe"
ckpt_from="last"
text_image_ratio=0.5
guidance=2.5
gpu_id=1
alpha=0.6

CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
fMRI2image.py \
--model_name $model_name --ckpt_from $ckpt_from \
--pool_type max \
--subj_load $subj_load --subj_test $subj_test \
--text_image_ratio $text_image_ratio --guidance $guidance \
--recons_per_sample 10 \
--alpha $alpha \

results_path="/data1/ruijie/train_logs/moe/recon_on_subj_1_$alpha"

CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
evaluate.py --results_path $results_path
