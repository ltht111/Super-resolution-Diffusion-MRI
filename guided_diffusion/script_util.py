import argparse
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unets import SuperResModel, SuperResUNetb2t1, SuperResUNetb2
from .unets_test import SuperResUNetd6t1
from .unet_35t import SuperResUNet35t

def sr_create_model_and_diffusion(args):
    """
    Create a super-resolution model and diffusion instance.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        tuple: Tuple containing the model and diffusion instances.
    """
    if args.b2:
        model = sr_create_unetb2(
            args.image_size,
            args.in_channel,
            args.num_channels,
            args.num_res_blocks,
            learn_sigma=args.learn_sigma,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=args.num_heads_upsample,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            resblock_updown=args.resblock_updown,
            use_fp16=args.use_fp16,
            t1 = args.t1
        )
    else:
        model = sr_create_model(
            args.image_size,
            args.in_channel,
            args.num_channels,
            args.num_res_blocks,
            learn_sigma=args.learn_sigma,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=args.num_heads_upsample,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            resblock_updown=args.resblock_updown,
            use_fp16=args.use_fp16,
        )
    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing=args.timestep_respacing,
        t1 = args.t1,
        b2 = args.b2
    )
    return model, diffusion

def sr_create_model(
        image_size,
        in_channel,
        num_channels,
        num_res_blocks,
        learn_sigma,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
        resblock_updown,
        use_fp16,
):
    channel_mult = (1, 1, 2, 2, 3, 3)

    attention_ds = [8, 16, 32]

    return SuperResUNet35t(
        image_size=image_size,
        in_channels=in_channel,
        model_channels=num_channels,
        out_channels=(in_channel if not learn_sigma else in_channel * 2), # 3 volume DTI
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )


def sr_create_unetb2(
        image_size,
        in_channel,
        num_channels,
        num_res_blocks,
        learn_sigma,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
        resblock_updown,
        use_fp16,
        t1
):
    channel_mult = (1, 1, 2, 2, 3, 3)

    attention_ds = [8, 16, 32]
    if t1:
        return SuperResUNetb2t1(
            image_size=image_size,
            in_channels=in_channel,
            model_channels=num_channels,
            out_channels=(in_channel if not learn_sigma else in_channel * 2 * 2), # 2b volume DTI
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_fp16=use_fp16,
        )
    else:
        return SuperResUNetb2(
            image_size=image_size,
            in_channels=in_channel,
            model_channels=num_channels,
            out_channels=(in_channel if not learn_sigma else in_channel * 2 * 2), # 2b volume DTI
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_fp16=use_fp16,
        )

def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
        t1=True,
        b2=True
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        t1=t1,
        b2=b2
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

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
