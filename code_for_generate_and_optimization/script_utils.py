import argparse
import torchvision
import torch.nn.functional as F

from unet import UNet
from diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data

def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        schedule="linear",
        schedule_low=1e-4, 
        schedule_high=0.02, 
        loss_type="l2",
        use_labels=False,

        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),

        ema_decay=0.9999,
        ema_update_rate=1,
        out_init_conv_padding=1,
        use_large_model=True,
        vocab_path="/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/data/vocab/vocab_1MER.txt",
        ernie_model_path="/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final",
        embedding_loss_weight=0.05,
        embedding_loss_steps=50,
        k_mer=1,
        max_seq_len=45,
    )

    return defaults

def get_diffusion_from_args(args):
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    unet_feature_dim = args.base_channels * max(args.channel_mults)
    num_classes = None
    print(f"\n{'='*60}")
    print("Model Configuration:")
    print(f"{'='*60}")
    print(f"base_channels: {args.base_channels}")
    print(f"channel_mults: {args.channel_mults}")
    print(f"bottleneck characteristic dimension {unet_feature_dim}")
    print(f"use_labels: {args.use_labels}")
    print(f"num_classes: {num_classes}") 
    print(f"{'='*60}\n")

    model = UNet(
        img_channels=1,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        num_res_blocks=args.num_res_blocks,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,
        num_classes=num_classes,
        initial_pad=0,
        out_init_conv_padding=args.out_init_conv_padding,
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    diffusion = GaussianDiffusion(
        model=model,
        img_size=(4, 44),
        img_channels=1,
        num_classes=num_classes,
        betas=betas,
        loss_type=args.loss_type,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        
        use_large_model=args.use_large_model,
        vocab_path=args.vocab_path if args.use_large_model else None,
        ernie_model_path=args.ernie_model_path if args.use_large_model else None,
        embedding_loss_weight=args.embedding_loss_weight,
        embedding_loss_steps=args.embedding_loss_steps,
        k_mer=args.k_mer,
        max_seq_len=args.max_seq_len,
        
        unet_feature_dim=unet_feature_dim if args.use_large_model else None,
        ernie_hidden_dim=768,
    )

    return diffusion


