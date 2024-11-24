# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from cog import BasePredictor, Input, Path
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer


from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.utils.conditioning_method import ConditioningMethod
from inference import (
    seed_everething,
    load_vae,
    load_unet,
    load_scheduler,
    load_image_to_tensor_with_resize_and_crop,
    calculate_padding,
)


MODEL_CACHE = "model_cache"
MODEL_URL = f"https://weights.replicate.delivery/default/Lightricks/LTX-Video/{MODEL_CACHE}.tar"

os.environ.update(
    {
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": MODEL_CACHE,
        "TORCH_HOME": MODEL_CACHE,
        "HF_DATASETS_CACHE": MODEL_CACHE,
        "TRANSFORMERS_CACHE": MODEL_CACHE,
        "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
    }
)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        vae = load_vae(Path(f"{MODEL_CACHE}/Lightricks/LTX-Video/vae"))
        unet = load_unet(Path(f"{MODEL_CACHE}/Lightricks/LTX-Video/unet"))
        scheduler = load_scheduler(
            Path(f"{MODEL_CACHE}/Lightricks/LTX-Video/scheduler")
        )
        patchifier = SymmetricPatchifier(patch_size=1)
        text_encoder = T5EncoderModel.from_pretrained(
            f"{MODEL_CACHE}/PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
        ).to("cuda")
        tokenizer = T5Tokenizer.from_pretrained(
            f"{MODEL_CACHE}/PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
        )
        unet = unet.to(torch.bfloat16)

        # Use submodels for the pipeline
        submodel_dict = {
            "transformer": unet,
            "patchifier": patchifier,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "vae": vae,
        }

        self.pipeline = LTXVideoPipeline(**submodel_dict).to("cuda")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="The waves crash against the jagged rocks of the shoreline. The waves crash against the jagged rocks of the shoreline, sending spray high into the air.The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon.",
        ),
        negative_prompt: str = Input(
            description="Negative prompt for undesired features",
            default="worst quality, inconsistent motion, blurry, jittery, distorted",
        ),
        image: Path = Input(
            description="Optional, input image for image-to-video generation",
            default=None,
        ),
        width: int = Input(
            description="Width of the output video frames. Optional if an input image provided",
            default=704,
            le=1280,
        ),
        height: int = Input(
            description="Height of the output video frames. Optional if an input image provided",
            default=480,
            le=720,
        ),
        num_frames: int = Input(
            description="Number of frames to generate in the output video",
            default=121,
            le=257,
        ),
        frame_rate: int = Input(
            description="Frame rate for the output video",
            default=25,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, default=40
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=3
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        seed_everething(seed)

        media_items_prepad = None
        if image is not None:
            media_items_prepad = load_image_to_tensor_with_resize_and_crop(
                str(image), height, width
            )

        # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

        padding = calculate_padding(height, width, height_padded, width_padded)

        media_items = None
        if media_items_prepad is not None:
            media_items = F.pad(
                media_items_prepad, padding, mode="constant", value=-1
            )  # -1 is the value for padding since the image is normalized to -1, 1

        sample = {
            "prompt": prompt,
            "prompt_attention_mask": None,
            "negative_prompt": negative_prompt,
            "negative_prompt_attention_mask": None,
            "media_items": media_items,
        }

        generator = torch.Generator(device="cuda").manual_seed(seed)

        images = self.pipeline(
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=frame_rate,
            **sample,
            is_video=True,
            vae_per_channel_normalize=True,
            conditioning_method=(
                ConditioningMethod.FIRST_FRAME
                if media_items is not None
                else ConditioningMethod.UNCONDITIONAL
            ),
            mixed_precision=True,
        ).images

        # Crop the padded images to the desired resolution and number of frames
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom
        pad_right = -pad_right
        if pad_bottom == 0:
            pad_bottom = images.shape[3]
        if pad_right == 0:
            pad_right = images.shape[4]
        images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]

        for i in range(images.shape[0]):
            # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
            video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
            # Unnormalizing images to [0, 255] range
            video_np = (video_np * 255).astype(np.uint8)
            height, width = video_np.shape[1:3]
            # In case a single image is generated
            if video_np.shape[0] == 1:
                output_filename = "/tmp/out.png"
                imageio.imwrite(output_filename, video_np[0])
            else:
                output_filename = "/tmp/out.mp4"
                # Write video
                with imageio.get_writer(output_filename, fps=frame_rate) as video:
                    for frame in video_np:
                        video.append_data(frame)

        return Path(output_filename)
