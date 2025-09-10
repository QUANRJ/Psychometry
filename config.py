import argparse


parser = argparse.ArgumentParser(description="Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
)
parser.add_argument(
    "--data_path", type=str, default="/data1/NSD_dataset",
)
parser.add_argument(
    "--subj_list",type=int, default=None, choices=[1,2,3,4,5,6,7,8], nargs='+',
)
parser.add_argument(
    "--batch_size", type=int, default=50,
)
parser.add_argument(
    "--val_batch_size", type=int, default=50,
)
parser.add_argument(
    "--clip_variant",type=str,default="ViT-L/14",choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"],
)
parser.add_argument(
    "--resume",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--resume_id",type=str,default=None,
)
parser.add_argument(
    "--load_from",type=str,default=None,
)
parser.add_argument(
    "--norm_embs",action=argparse.BooleanOptionalAction,default=True,
    help="Do l2-norming of CLIP embeddings",
)
parser.add_argument(
    "--use_image_aug",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--num_epochs",type=int,default=500,
)
parser.add_argument(
    "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
)
parser.add_argument(
    "--ckpt_interval",type=int,default=10,
)
parser.add_argument(
    "--eval_interval",type=int,default=10,
)
parser.add_argument(
    "--seed",type=int,default=1010,
)
parser.add_argument(
    "--num_workers",type=int,default=4,
)
parser.add_argument(
    "--max_lr",type=float,default=3e-4,
)
parser.add_argument(
    "--pool_num", type=int, default=8192,
)
parser.add_argument(
    "--pool_type", type=str, default='max',
)
parser.add_argument(
    "--mse_mult", type=float, default=1e4,
)
parser.add_argument(
    "--rec_mult", type=float, default=0,
)
parser.add_argument(
    "--cyc_mult", type=float, default=0,
)
parser.add_argument(
    "--length", type=int, default=None,
)
parser.add_argument(
    "--autoencoder_name", type=str, default='model_encoder',
)
parser.add_argument(
    "--subj_load",type=int, default=None, choices=[1,2,5,7], nargs='+',
)
parser.add_argument(
    "--subj_test",type=int, default=1, choices=[1,2,5,7],
)
parser.add_argument(
    "--samples",type=int, default=None, nargs='+',
)
parser.add_argument(
    "--img2img_strength",type=float, default=.85,
)
parser.add_argument(
    "--guidance_scale",type=float, default=3.5,
)
parser.add_argument(
    "-num_inference_steps",type=int, default=20,
)
parser.add_argument(
    "--recons_per_sample", type=int, default=16,
)
parser.add_argument(
    "--plotting", action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--vd_cache_dir", type=str, default='./weights',
)
parser.add_argument(
    "--gpu_id", type=int, default=0,
)
parser.add_argument(
    "--ckpt_from", type=str, default='last',
)
parser.add_argument(
    "--text_image_ratio", type=float, default=0.5,
)
parser.add_argument(
    "--only_embeddings", action=argparse.BooleanOptionalAction, default=False,
)
parser.add_argument(
    "--synthesis", action=argparse.BooleanOptionalAction, default=False,
)
parser.add_argument(
    "--verbose", action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--results_path", type=str, default=None,
)
parser.add_argument(
    "--output_dir", type=str, default='/data2/temp',
)
parser.add_argument(
    "--use_ecphory", action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--alpha", type=float, default=None,
)

args = parser.parse_args()