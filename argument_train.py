import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='representational-generation')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--target-model', type=str, default='SD_14') #required=True, 
    parser.add_argument('--refer-dataset', type=str, default='fairface', choices=['fairface', 'stable_bias_i'])
    parser.add_argument('--mpr-onehot', default=False, action='store_true', help='onehot group estimation')
    parser.add_argument('--functionclass', type=str, default='linear', help='functionclass for mpr')
    
    # Hyperparameters for training
    parser.add_argument('--train', default=True, action='store_false', help='train the model')
    parser.add_argument('--trainer', default='finetuning', type=str, help='choose the trainer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', default=0.0001, type=float, help='weight_decay')    
    parser.add_argument('--n-iters', default=50, type=int, help='total iterations')
    parser.add_argument('--trainer-group', type=str, nargs='+', default=['gender','age','race'])    
    parser.add_argument('--training-dataset', type=str, default=None)    
    parser.add_argument('--race-reduce', default=False, action='store_true', help='reduce race')
    parser.add_argument('--bal-sampling', default=False, action='store_true')
    parser.add_argument('--eniac', default=False, action='store_true')

    parser.add_argument('--iterations', type=int, default=10000, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--method', type=str, default=None, help='method to mitigate unfairness')
    parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer')
    parser.add_argument('--lr-scheduler', type=str, default='cosine', help='scheduler')
    parser.add_argument('--temp', type=float, default=1, help='learning rate')

    # hyperparameters for each method
    parser.add_argument('--lamb', type=float, default=0.5, help='regularization strength')

    # Info for result file names 
    parser.add_argument('--date', type=str, default='default', help='date when to save')
    parser.add_argument('--save-dir', type=str, default='results/', help='directory to save the results')

    # for finetuning
    parser.add_argument('--finetuning-ver', type=str, default='ver1',choices = ['ver1','ver2','ver3'])
    parser.add_argument('--normalize', default=False, action='store_true', help='normalization for x')
    
    parser.add_argument('--mpr_num_batches', type=int, default=4, help='the number of batches for computing MPRs')
    parser.add_argument('--n-cs', type=int, default=32, help='the number of constraints')
    parser.add_argument('--vision-encoder', type=str, default='CLIP',choices = ['BLIP', 'CLIP', 'PATHS'])
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        default=False,
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_unet",
        action="store_true",
        default=False,
        help="Whether to train unet. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=20,
        help=(
            "Save a temporary checkpoint every X steps. "
            "The purpose of these checkpoints is to easily resume training "
            "when some error occurs during training."
        )
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=(
            "Max number of temporary checkpoints to store. "
            "The oldest ones will be deleted when new checkpoints are saved."),
    )
    parser.add_argument(
        "--checkpointing_steps_long",
        type=int,
        default=200,
        help=(
            "Save a checkpoint every Y steps. "
            "These checkpoints will not be deleted. They are used for final evaluation. "
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="provide the checkpoint path to resume from checkpoint",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=50,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        '--train_plot_every_n_iter', 
        help="plot training stats every n iteration", 
        type=int, 
        default=20
        )
    parser.add_argument(
        '--evaluate_every_n_iter', 
        help="evaluate every n iteration", 
        type=int,
        default=200
        )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help='only `"wandb"` is supported',
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        '--guidance_scale', 
        help="diffusion model text guidance scale", 
        type=float, 
        default=7.5
        )
    parser.add_argument(
        '--EMA_decay', 
        help="decay coefficient for EMA",
        type=float,
        default=0.996
        )

    # loss weight
    parser.add_argument(
        '--weight_loss_img', 
        default=8,
        help="weight for the image semantics preserving loss", 
        type=float, 
    )
    parser.add_argument(
        '--weight_loss_face', 
        default=1,
        help="weight for the face realism preserving loss", 
        type=float, 
    )
    parser.add_argument(
        '--uncertainty_threshold', 
        help="the uncertainty threshold used in distributional alignment loss", 
        type=float, 
        default=0.2
        )
    parser.add_argument('--factor1', help="train, val, test batch size", type=float, default=0.2)
    parser.add_argument('--factor2', help="train, val, test batch size", type=float, default=0.2)

    # batch size, properly set to max out GPU
    parser.add_argument(
        '--train_images_per_prompt_GPU', 
        help=(
            "number of images generated for a prompt per GPU during training. "
            "These images are used as a batch for distributional alignment."
        ), 
        type=int, 
        default=8,
        )
    parser.add_argument(
        '--train_GPU_batch_size', 
        help="training batch size in every GPU", 
        type=int, 
        default=4
        )
    parser.add_argument(
        '--val_images_per_prompt_GPU', 
        help=(
            "number of images generated for a prompt per GPU during validation. "
            "These images are used to measure bias."
        ),
        type=int, 
        default=100
        )
    parser.add_argument(
        '--val_GPU_batch_size', 
        help="validation batch size in every GPU", 
        type=int, 
        default=8
        )    


    # experiment input and output paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="logs will be saved to args.output_dir/args.logging_dir",
    )
    parser.add_argument(
        "--prompt_occupation_path",
        type=str,
        default="datasets/data_for_finetuning/1-prompts/occupation.json",
        help="prompt template, and occupations for train and val",
    )
    parser.add_argument(
        '--classifier_weight_path', 
        default="datasets/data_for_finetuning/2-trained-classifiers/CelebA_MobileNetLarge_08060852/epoch=9-step=12660_MobileNetLarge.pt",
        help="pre-trained classifer that predicts binary gender", 
        type=str,
        required=False, 
    )
    parser.add_argument(
        '--face_feats_path', 
        help="external face feats, used for the face realism preserving loss", 
        type=str, 
        default="datasets/data_for_finetuning/3-face-features/CelebA_MobileNetLarge_08240859/face_feats.pkl"
        )
    # parser.add_argument(
    #     '--aligned_face_gender_model_path', 
    #     help="train, val, test batch size", 
    #     type=str, 
    #     default="../data/3-face-features/CelebA_MobileNetLarge_08240859/epoch=9-step=6330_MobileNetLarge.pt"
    #     )
    parser.add_argument('--opensphere_config', help="train, val, test batch size", type=str, default="datasets/data_for_finetuning/4-opensphere_checkpoints/opensphere_checkpoints/20220424_210641/config.yml")
    parser.add_argument('--opensphere_model_path', help="train, val, test batch size", type=str, default="datasets/data_for_finetuning/4-opensphere_checkpoints/opensphere_checkpoints/20220424_210641/models/backbone_100000.pth")

    # learning related settings
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power", 
        type=float, 
        default=1.0, 
        help="Power factor of the polynomial scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=100.0, type=float, help="Max gradient norm.")

    # settings that should not be changed
    # we didn't experiment with other values
    parser.add_argument(
        "--img_size_small",
        type=int,
        default=224,
        help="For some operations, images will be resized to this size for more efficient processing",
    )
    parser.add_argument(
        "--size_face",
        type=int,
        default=224,
        help="faces will be resized to this size",
    )
    parser.add_argument(
        "--size_aligned_face",
        type=int,
        default=112,
        help="aligned faces will be resized to this size",
    )
    parser.add_argument('--face_gender_confidence_level', help="train, val, test batch size", type=float, default=0.9)

    # passed directly by accelerate
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
        )
    
    # config file
    parser.add_argument("--config", help="config file", type=str, default=None)


    args = parser.parse_args()
    return args
