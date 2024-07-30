# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from tokenize_anything.modeling import ImageTokenizer
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack

from typing import Optional, Tuple, List

# from .utils.transforms import ResizeLongestSide


class TapPredictor:
    def __init__(
        self,
        tap_model: ImageTokenizer,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = tap_model
        self.image_size = tap_model.image_encoder.image_size
        # self.transform = ResizeLongestSide(self.image_size)
        self.device = tap_model.image_encoder.device
        self.dtype = tap_model.image_encoder.dtype
        self.inputs = None
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).to(self.device)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).to(self.device)
        self.reset_image()
        
    @torch.no_grad()
    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        # input_image = self.transform.apply_image(image)
        # input_image_torch = torch.as_tensor(input_image, device=self.device)
        # input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        img_list, self.img_scales = im_rescale(image, scales=[1024], max_size=1024)
        self.input_size, self.original_size = img_list[0].shape, image.shape[:2]
        self.img_batch = im_vstack(img_list, fill_value=self.model.pixel_mean_value, size=(self.image_size, self.image_size))
        self.inputs = self.model.get_inputs({"img": self.img_batch})
        self.inputs.update(self.model.get_features(self.inputs))
        self.features = self.inputs['img_embeds'].squeeze(1)
        self.is_image_set = True

        # self.set_torch_image(image, image.shape[:2])

    # @torch.no_grad()
    # def set_torch_image(
    #     self,
    #     transformed_image: torch.Tensor,
    #     original_image_size: Tuple[int, ...],
    # ) -> None:
    #     """
    #     Calculates the image embeddings for the provided image, allowing
    #     masks to be predicted with the 'predict' method. Expects the input
    #     image to be already transformed to the format expected by the model.

    #     Arguments:
    #       transformed_image (torch.Tensor): The input image, with shape
    #         1x3xHxW, which has been transformed with ResizeLongestSide.
    #       original_image_size (tuple(int, int)): The size of the image
    #         before transformation, in (H, W) format.
    #     """
    #     assert (
    #         len(transformed_image.shape) == 4
    #         and transformed_image.shape[1] == 3
    #         and max(*transformed_image.shape[2:]) == self.model.image_encoder.image_size
    #     ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.image_size}."
    #     self.reset_image()

    #     self.original_size = original_image_size
    #     self.input_size = tuple(transformed_image.shape[-2:])
    #     self.input_image = self.preprocess(transformed_image)
    #     self.inputs = self.model.get_inputs({"img": self.input_image})
    #     self.inputs.update(self.model.get_features(self.inputs))
    #     self.features = self.inputs['img_embeds'].squeeze(1)
    #     # self.features = self.model.image_encoder(input_image)
    #     self.is_image_set = True

    def predict(
        self,
        points: Optional[np.ndarray] = None,
        # box: Optional[np.ndarray] = None,
        # mask_input: Optional[np.ndarray] = None,
        # multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        masks, iou_predictions, low_res_masks, captions, sem_tokens = self.predict_torch(points, return_logits)
        masks_np = masks.detach().cpu().numpy()
        iou_predictions_np = iou_predictions.detach().cpu().numpy()
        low_res_masks_np = low_res_masks.detach().cpu().numpy()
        sem_tokens_np = sem_tokens.detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np, captions, sem_tokens_np
            
    @torch.no_grad()
    def predict_torch(
        self,
        points: Optional[torch.Tensor],
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # scale_w = self.image_size / self.original_size[1]
        # scale_h = self.image_size / self.original_size[0]
        points[:, :, :2] *= np.array(self.img_scales, "float32")
        # points[:, :, 0] *= scale_w
        # points[:, :, 1] *= scale_h
        self.inputs['points'] = points
        # assert self.inputs is not None, 'inputs is None, please input image first.'
        outputs = self.model.get_outputs(self.inputs)
        # Select final mask.
        iou_score, mask_pred = outputs["iou_pred"], outputs["mask_pred"]
        iou_score[:, 0] -= 1000.0  # Penalize the score of boundary boxes.
        mask_index = torch.arange(iou_score.shape[0]), iou_score.argmax(1)
        # mask_index = torch.arange(iou_score.shape[0]), 3
        # mask_index = 1

        # Upscale masks to the original image resolution.
        iou_predictions, low_res_masks = iou_score[mask_index], mask_pred[mask_index]
        masks = self.model.upscale_masks(low_res_masks[:, None], self.img_batch.shape[1:-1])
        masks = masks[..., : self.input_size[0], : self.input_size[1]]
        if return_logits:
          masks = self.model.upscale_masks(masks, self.original_size)
        else:
          masks = self.model.upscale_masks(masks, self.original_size).gt(0)

        # Predict concepts and generate captions.
        sem_tokens, sem_embeds = outputs["sem_tokens"], outputs["sem_embeds"]
        sem_tokens = sem_tokens[mask_index]
        # concepts, scores = self.model.predict_concept(sem_embeds[mask_index])
        captions = self.model.generate_text(sem_tokens[:, None, :])
        
        return masks, iou_predictions, low_res_masks, captions, sem_tokens
        # return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    # @property
    # def device(self) -> torch.device:
    #     return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.img_scales = None
        self.img_batch = None
        self.inputs = None
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
    
    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = torch.nn.functional.pad(x, (0, padw, 0, padh))
        return x