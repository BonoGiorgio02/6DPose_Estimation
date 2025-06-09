import torch
from torch import nn

class PointProjector(nn.Module):
    """
    Project 3D points in the image coordinates by using camera intrinsics. It assumes points to be in camera coordinate
    """
    def __init__(self, fx=572.4114, fy=573.57043, cx=325.2611, cy=242.04899):
        super().__init__()
        # fixed camera intrinsics matrix
        self.register_buffer('camera_intrinsics', torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ]))

    def forward(self, points_3d, bbox_base, padding, filtering=True):
        """
        Args:
            points_3d: [N, 3] 3D coordinates of points in camera system
            bbox_base: [4] (x_min, y_min, crop_width, crop_height)
            padding: [4] (pad_left, pad_right, pad_top, pad_bottom)
        Returns:
            pixel_coords: [N, 2] pixel coordinates (u, v) to the cropped and padded image
            valid_mask: [N] mask for valid points (inside cropped and padded image)
        """
        device = points_3d.device

        # extract bbox and padding parameters
        x_min, y_min, crop_width, crop_height = bbox_base.int()
        pad_left, pad_right, pad_top, pad_bottom = padding.int()

        # project to original image plane (640x480)
        points_2d_homo = torch.matmul(points_3d, self.camera_intrinsics.T)
        pixel_coords_original = points_2d_homo[:, :2] / (points_2d_homo[:, 2:3] + 1e-8)

        # transform to cropped image coordinates
        pixel_coords_crop = torch.zeros_like(pixel_coords_original)
        pixel_coords_crop[:, 0] = pixel_coords_original[:, 0] - x_min  # u coordinate
        pixel_coords_crop[:, 1] = pixel_coords_original[:, 1] - y_min  # v coordinate

        # transform to padded image coordinates
        pixel_coords_padded = torch.zeros_like(pixel_coords_crop)
        pixel_coords_padded[:, 0] = pixel_coords_crop[:, 0] + pad_left
        pixel_coords_padded[:, 1] = pixel_coords_crop[:, 1] + pad_top

        # valid mask checks
        # check depth
        if filtering:
            valid_depth = points_3d[:, 2] > 1e-8  # min depth

            # check if points are within original crop area
            valid_in_original_crop = (
                (pixel_coords_original[:, 0] >= x_min) &
                (pixel_coords_original[:, 0] < x_min + crop_width) &
                (pixel_coords_original[:, 1] >= y_min) &
                (pixel_coords_original[:, 1] < y_min + crop_height)
            )

            # check if points are within padded image bounds
            padded_width = crop_width + pad_left + pad_right
            padded_height = crop_height + pad_top + pad_bottom

            valid_in_padded = (
                (pixel_coords_padded[:, 0] >= 0) &
                (pixel_coords_padded[:, 0] < padded_width) &
                (pixel_coords_padded[:, 1] >= 0) &
                (pixel_coords_padded[:, 1] < padded_height)
            )

            valid_mask = valid_depth & valid_in_original_crop & valid_in_padded
        else:
            valid_mask = torch.ones(points_3d.shape[0], dtype=torch.bool, device=device)

        return pixel_coords_padded, valid_mask