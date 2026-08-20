"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import sys
from datetime import datetime
import argparse
import yaml
import numpy as np
import cv2
import nibabel as nib
from pathlib import Path

# from dataset.prepare_not1 import lrgraceset
from dataset.prepare_lrgrace import lrgraceset
from torch.utils.data import DataLoader
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_create_model_and_diffusion,
    add_dict_to_argparser,
)

def main():
    args = create_argparser().parse_args()

    # These explicit CLI options are used by sample_nifti.sh so the standalone
    # folder never depends on paths from the original source tree.
    if args.input_root:
        args.lr_data_dir = args.input_root
    if args.checkpoint:
        args.model_path = args.checkpoint

    direction = getattr(args, "direction", "cor").lower()
    if direction not in {"axial", "cor", "sag"}:
        raise ValueError("direction must be one of: axial, cor, sag")
    if not args.grace or not args.b2 or not args.t1:
        raise ValueError(
            "This standalone entry point expects grace=true, b2=true and t1=true."
        )

    input_root = Path(args.lr_data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    model_path = Path(resolve_model_path(args.model_path, direction)).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    patients = sorted(
        path for path in input_root.iterdir()
        if path.is_dir() and (path / "3D.nii.gz").is_file()
    )
    if not patients:
        raise FileNotFoundError(
            f"No case containing 3D.nii.gz was found under: {input_root}"
        )
    for case_dir in patients:
        for required_name in ("T1_brain.nii.gz", "T1_brain_mask.nii.gz"):
            required_path = case_dir / required_name
            if not required_path.is_file():
                raise FileNotFoundError(f"Missing required input: {required_path}")
        dwi_image = nib.load(str(case_dir / "3D.nii.gz"))
        t1_image = nib.load(str(case_dir / "T1_brain.nii.gz"))
        mask_image = nib.load(str(case_dir / "T1_brain_mask.nii.gz"))
        if len(dwi_image.shape) != 4 or dwi_image.shape[-1] != 31:
            raise ValueError(
                f"DWI must have shape (x,y,z,31), got {dwi_image.shape}: "
                f"{case_dir / '3D.nii.gz'}"
            )
        if t1_image.shape != dwi_image.shape[:3] or mask_image.shape != dwi_image.shape[:3]:
            raise ValueError(
                f"DWI/T1/mask spatial shapes must match in {case_dir}: "
                f"DWI={dwi_image.shape[:3]}, T1={t1_image.shape}, mask={mask_image.shape}"
            )
    
    print(args)
    sys.stdout.flush()
    logger.configure()

    logger.log("creating model...")
    sys.stdout.flush()
    

    logger.log(f"sampling direction: {direction}")
    logger.log(f"input root: {input_root}")
    logger.log(f"output root: {output_root}")
    logger.log(f"checkpoint: {model_path}")
    date = datetime.now().strftime('%y%m%d_%H')
    run_output = output_root / date
    run_output.mkdir(parents=True, exist_ok=True)
    for case_dir in patients:
        case_name = case_dir.name
        # if dir == '50hz.nii':
        #     args.batch_size = 2
        # else:
        #     args.batch_size = 4
        grace_dataset = lrgraceset(case_dir / "3D.nii.gz")
        maxn_i, _ = grace_dataset.maxn
        b0_index_l = grace_dataset.b0_index_l
        b1000_index = grace_dataset.b1000_index
        head_affine = grace_dataset.head_affine

        model, diffusion = sr_create_model_and_diffusion(args)
    
        logger.log(f"loading checkpoint for {direction}: {model_path}")
        model.load_state_dict(
            dist_util.load_state_dict(str(model_path), map_location="cpu")
        )

        model.to(dist_util.dev())
        
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        logger.log("loading data...")
        
        grace_dataset.direc(direction)
        maskdata = load_grace_data(
            grace_dataset, batch_size=args.batch_size, t1=args.t1
        )

        logger.log("creating samples...")

        
        sample_path = run_output / case_name
        sample_path.mkdir(parents=True, exist_ok=True)

        # psnr_list, ssim_list = [], []
        plane_h, plane_w, slice_count, _ = grace_dataset.ir.shape
        volume_count = max(b0_index_l + b1000_index) + 1
        newsr_save = np.zeros(
            (plane_h, plane_w, slice_count, volume_count), dtype=np.float32
        )
        b0_sum = np.zeros((plane_h, plane_w, slice_count), dtype=np.float32)
        b0_count = np.zeros(slice_count, dtype=np.int32)
        sample_offset = 0
        directions_per_slice = len(b1000_index)

        for i, (model_kwargs, mask) in enumerate(maskdata):
            current_batch = int(model_kwargs['low_res'].shape[0])

            mask = mask.permute(0, 2, 3, 1).contiguous().cpu().numpy()

            lr = model_kwargs['low_res']
            lr = lr.permute(0,2,3,1)
            lr = lr.contiguous()
            lr = lr.cpu().numpy()

            t1 = model_kwargs['other']
            t1 = t1.permute(0,2,3,1)
            t1 = t1.contiguous()
            t1 = t1.cpu().numpy()

            model_kwargs = {k: (v.to(dist_util.dev()) if v is not None else None) for k, v in model_kwargs.items()}

            # Choose the appropriate sample function based on the sampling method
            sample_fn = diffusion.ddim_sample_loop if args.sampling_method == 'ddim' else \
                (diffusion.dpm_solver_sample_loop if args.sampling_method == 'dpm++' else diffusion.p_sample_loop)

            sample = sample_fn(
                model,
                (current_batch, args.in_channel*2, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            sample = sample.cpu().numpy()
            sample = (sample.T * mask.T).T
            
            for b1 in range(sample.shape[0]):
                sample_img = sample[b1, ...].clip(0,1)
                t1_img = t1[b1, ...].clip(0,1)
                lr_img = lr[b1, ...].clip(0,1)
                b0 = (sample_img[:,:,0])#*np.mean(maxn_i[b0_index_l])
                b1000 = (sample_img[:,:,1])#*maxn_i[b1000_index[b1]]

                global_index = sample_offset + b1
                slice_index = global_index // directions_per_slice
                direction_index = global_index % directions_per_slice
                volume_index = b1000_index[direction_index]
                newsr_save[:, :, slice_index, volume_index] = (
                    b1000 * maxn_i[volume_index]
                )
                b0_sum[:, :, slice_index] += b0 * np.mean(maxn_i[b0_index_l])
                b0_count[slice_index] += 1
                
                # Optional PNG debugging output is disabled. This does not
                # affect reconstruction or the final NIfTI result.
                # if i > 50 and i < 70:
                #     cv2.imwrite(
                #         str(sample_path / f'{i}_1b{b1}_sample.png'),
                #         (sample_img[:, :, 1].clip(0, 1) * 255.0).round().astype(np.uint8),
                #     )
                #     cv2.imwrite(
                #         str(sample_path / f'{i}_1b{b1}_lr.png'),
                #         (lr_img[:, :, 1].clip(0, 1) * 255.0).round().astype(np.uint8),
                #     )
                #     cv2.imwrite(
                #         str(sample_path / f'{i}_1b{b1}_t1.png'),
                #         (t1_img.clip(0, 1) * 255.0).round().astype(np.uint8),
                #     )
            sample_offset += current_batch
            print(i)
            sys.stdout.flush()

        if sample_offset != grace_dataset.data_len:
            raise RuntimeError(
                f"sample count mismatch: got {sample_offset}, "
                f"expected {grace_dataset.data_len}"
            )
        for slice_index in range(slice_count):
            if b0_count[slice_index] == 0:
                continue
            b0_average = b0_sum[:, :, slice_index] / b0_count[slice_index]
            for volume_index in b0_index_l:
                newsr_save[:, :, slice_index, volume_index] = b0_average

        # First undo the selected slicing direction, then retain the legacy
        # output layout used by this project so axial/cor/sag results remain
        # aligned with previously generated NIfTI files and the SR3 outputs.
        newsr_save = grace_dataset.restore_volume(newsr_save, direction)
        newsr_save = np.transpose(np.flip(newsr_save, axis=0), (2, 0, 1, 3))
        sr_nii = nib.Nifti1Image(newsr_save, head_affine[1], head_affine[0])
        output_file = run_output / f"{case_name}_2_{direction}.nii.gz"
        nib.save(sr_nii, str(output_file))
        logger.log(f"saved {direction} result: {output_file}, shape={newsr_save.shape}")

    logger.log("sampling complete")


def load_grace_data(grace_dataset, batch_size, t1):
    grace_loader = DataLoader(
        grace_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  #lht changed
        drop_last=False,
        pin_memory=True
    )
    # Iterate over the data loader and yield high-resolution MRIs and model keyword arguments
    if t1:
        for lr_data, other_data, mask_data in grace_loader:
            model_kwargs = {"low_res": lr_data, "other": other_data}
            yield model_kwargs, mask_data
    else:
        for lr_data, mask_data in grace_loader:
            # print(hr_data.shape)
            model_kwargs = {"low_res": lr_data, "other": None}
            yield model_kwargs, mask_data


def resolve_model_path(model_path, direction):
    """Accept one shared checkpoint or a direction-to-checkpoint mapping."""
    if isinstance(model_path, str):
        return model_path
    if not isinstance(model_path, dict) or not model_path:
        raise ValueError("model_path must be a checkpoint string or a non-empty mapping")
    if direction in model_path and model_path[direction]:
        return model_path[direction]

    available = [(key, value) for key, value in model_path.items() if value]
    if len(available) == 1:
        key, value = available[0]
        logger.log(
            f"model_path has no '{direction}' key; using the only configured "
            f"checkpoint under '{key}'"
        )
        return value
    raise KeyError(
        f"model_path has no checkpoint for direction '{direction}'. "
        f"Available keys: {list(model_path.keys())}"
    )


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML configuration file")
    # Read the config path first but leave YAML-defined overrides such as
    # ``--direction sag`` for the final parse in main().
    args, _ = parser.parse_known_args()
    
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        
    add_dict_to_argparser(parser, config)
    parser.add_argument(
        "--input-root",
        default=None,
        help="Override lr_data_dir from YAML. Each child directory is one case.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override model_path from YAML with one checkpoint file.",
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Directory in which timestamped test results are written.",
    )
    return parser


if __name__ == "__main__":
    main()
