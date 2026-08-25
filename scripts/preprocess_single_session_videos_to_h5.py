#!/usr/bin/env python3
"""
Preprocess a single session's video files into an HDF5 file.

Reads transform parameters from a JSON config file, applies them to all .mp4
files in the data folder, and writes the result to a single HDF5 file.

Usage examples:

    # Basic — uses all values from the config file:
    python preprocess_single_session_videos_to_h5.py \
        --config ../configs/ae_config.json \
        --h5-filename session.h5 \
        --save-path /path/to/output/

    # Override a few crop parameters without editing the config:
    python preprocess_single_session_videos_to_h5.py \
        --config ../configs/ae_config.json \
        --h5-filename session.h5 \
        --save-path /path/to/output/ \
        --crop-top 10 --crop-left 200

    # Force overwrite an existing HDF5 file:
    python preprocess_single_session_videos_to_h5.py \
        --config ../configs/ae_config.json \
        --h5-filename session.h5 \
        --save-path /path/to/output/ \
        --overwrite

    # Use a per-session crop config (e.g. from configs/crop_configs/) to
    # overwrite config['data'], with any explicit --crop-* flags still
    # taking precedence over the session config:
    python preprocess_single_session_videos_to_h5.py \
        --config ../configs/ae_config.json \
        --session-config ../configs/crop_configs/kd115_twNew_20221206_115814.json \
        --h5-filename session.h5 \
        --save-path /path/to/output/
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Make the project's `src/` importable without installing the package.
# This script lives in <project>/scripts/, so we go one level up to find src/.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from preprocessing.video_preprocessing_utils import (
    convert_videos_to_hdf5,
    get_video_transforms,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a folder of .mp4 trial videos into a single HDF5 file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Required / main arguments ----------------------------------------
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON config file (e.g. configs/ae_config.json).",
    )
    parser.add_argument(
        "--session-config",
        type=str,
        default=None,
        help="Optional path to a per-session JSON config (e.g. from "
        "configs/crop_configs/) whose keys overwrite config['data']. "
        "Explicit --image-height/--crop-*/--data-path flags still take "
        "precedence over this file.",
    )
    parser.add_argument(
        "--h5-filename",
        type=str,
        required=True,
        help="Name of the output HDF5 file (e.g. session_data.h5).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory where the HDF5 file will be saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, overwrite the output file when it already exists.",
    )

    # ---- Optional overrides for config['data'] fields ---------------------
    # Defaults are None so we can distinguish "user supplied a value" from
    # "user wants the config default".
    data_group = parser.add_argument_group(
        "config['data'] overrides",
        "Override individual values from the config's 'data' section. "
        "Any value left unset keeps the original config value.",
    )
    data_group.add_argument("--image-height", type=int, default=None,
                            help="Target image height after resize.")
    data_group.add_argument("--image-width", type=int, default=None,
                            help="Target image width after resize.")
    data_group.add_argument("--crop-top", type=int, default=None,
                            help="Pixels to crop from the top.")
    data_group.add_argument("--crop-bottom", type=int, default=None,
                            help="Pixels to crop from the bottom.")
    data_group.add_argument("--crop-left", type=int, default=None,
                            help="Pixels to crop from the left.")
    data_group.add_argument("--crop-right", type=int, default=None,
                            help="Pixels to crop from the right.")
    data_group.add_argument("--data-path", type=str, default=None,
                            help="Path to the folder containing .mp4 videos.")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Load config -------------------------------------------------------
    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        sys.exit(f"Error: config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # ---- Apply per-session config overrides to config['data'] -------------
    # Overwrites config['data'] wholesale with the session config's keys;
    # any key not present in the session config keeps its original value.
    if args.session_config is not None:
        session_config_path = os.path.abspath(args.session_config)
        if not os.path.isfile(session_config_path):
            sys.exit(f"Error: session config file not found: {session_config_path}")

        with open(session_config_path, "r") as f:
            session_config = json.load(f)

        config["data"].update(session_config)

    # ---- Apply CLI overrides to config['data'] -----------------------------
    # Only overwrite keys that the user explicitly provided.
    cli_overrides = {
        "image_height": args.image_height,
        "image_width": args.image_width,
        "crop_top": args.crop_top,
        "crop_bottom": args.crop_bottom,
        "crop_left": args.crop_left,
        "crop_right": args.crop_right,
        "data_path": args.data_path,
    }

    for key, value in cli_overrides.items():
        if value is not None:
            config["data"][key] = value

    # ---- Check for existing output file ------------------------------------
    save_path = os.path.abspath(args.save_path)
    output_file = os.path.join(save_path, args.h5_filename)

    if os.path.isfile(output_file) and not args.overwrite:
        print(
            f"Output file already exists: {output_file}\n"
            "Use --overwrite to replace it."
        )
        sys.exit(0)

    # ---- Build transforms and run conversion -------------------------------
    print("Effective config['data']:")
    for k, v in config["data"].items():
        print(f"  {k}: {v}")
    print()

    transforms = get_video_transforms(config)

    os.makedirs(save_path, exist_ok=True)

    convert_videos_to_hdf5(
        video_folder=config["data"]["data_path"],
        transform=transforms,
        h5_filename=args.h5_filename,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
