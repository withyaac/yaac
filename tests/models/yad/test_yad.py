"""Tests for the YAD model."""

import pytest
import torch
import numpy as np

from yaac.models.yad.yad import YAD, make_yad, _make_anchor_points, _get_pairwise_distance_points_bboxes, _get_anchor_gt_matches, _get_bbox_offsets


def test_make_yad_shapetype():
    """Test that make_yad creates a model with correct type and structure."""
    model = make_yad(num_classes=80)
    assert isinstance(model, YAD)
    assert isinstance(model, torch.nn.Module)


def test_yad_forward_shapetype():
    """Test that the forward pass outputs have correct shapes and types."""
    num_classes = 80
    model = make_yad(num_classes=num_classes)
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)
    bbox_out, cls_out, objectness_out = model(image)

    # Check types
    assert isinstance(bbox_out, torch.Tensor)
    assert isinstance(cls_out, torch.Tensor)
    assert isinstance(objectness_out, torch.Tensor)

    # Check shapes
    assert bbox_out.shape[0] == batch_size
    assert cls_out.shape[0] == batch_size
    assert objectness_out.shape[0] == batch_size
    assert cls_out.shape[1] == num_classes
    assert bbox_out.shape[1] == 4
    assert objectness_out.shape[1] == 1


def test_make_anchor_points_values():
    """Test that anchor points are placed correctly in normalized coordinates."""
    # Input parameters - using a small 3x3 grid for easy verification  
    feature_map_height = 3
    feature_map_width = 3

    # Expected outputs - hard-coded for a 3x3 grid
    expected_points = torch.tensor([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],  # First row
        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],  # Middle row
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],  # Last row
    ])

    # Call function under test
    anchor_points = _make_anchor_points(
        feature_map_height=feature_map_height,
        feature_map_width=feature_map_width,
    )

    # Verify outputs
    assert anchor_points.shape == (9, 2)
    assert torch.allclose(anchor_points, expected_points)


def test_get_pairwise_distance_points_bboxes_values():
    """Test that pairwise distances between points and boxes are computed correctly.
    
    Example:
        Points: 3 points in a line
        Boxes: 2 boxes
        - Box 1: Small box in the middle (0.4, 0.4) to (0.6, 0.6)
        - Box 2: Large box covering most of the space (0.1, 0.1) to (0.9, 0.9)
        
        Expected distances:
        - Points outside boxes: distance = 1.0
        - Points inside boxes: distance = normalized distance to center
        - Point at box center: distance = 0.0
        - Point at box corner: distance = 1.0
    """
    # Input points - 3 points in a line
    points = torch.tensor([
        [0.0, 0.5],  # Point 1: Outside both boxes
        [0.5, 0.5],  # Point 2: Center of both boxes
        [0.8, 0.5],  # Point 3: Inside box 2, outside box 1
    ])

    # Input boxes - 2 boxes
    boxes = torch.tensor([
        [0.4, 0.4, 0.6, 0.6],  # Box 1: Small box in middle
        [0.1, 0.1, 0.9, 0.9],  # Box 2: Large box
    ])

    # Expected distances
    expected_distances = torch.tensor([
        [1.0, 1.0],  # Point 1: Outside both boxes
        [0.0, 0.0],  # Point 2: Center of both boxes
        [1.0, 0.3 / np.sqrt(0.4**2 + 0.4**2)],  # Point 3: Outside box 1, halfway to corner of box 2
    ]).to(torch.float32)
    # explain 0.4 / np.sqrt(0.4**2 + 0.4**2
    # 0.4 is the distance from the center of the box to the corner for each coord
    # np.sqrt(0.4**2 + 0.4**2) is the diagonal of the box
    # 0.3 is the distance from the center of the box to the point
    # 0.3 / np.sqrt(0.4**2 + 0.4**2) is the normalized distance to the corner
    

    # Call function under test
    distances = _get_pairwise_distance_points_bboxes(points, boxes)

    # Verify outputs
    assert distances.shape == (3, 2)
    assert torch.allclose(distances, expected_distances, atol=1e-6)


def test_get_anchor_gt_matches_values():
    """Test that anchor-to-gt matching works correctly with different distance thresholds.
    
    Example:
        Distances: 3 anchor points, 2 boxes
        - Anchor 1: [0.3, 0.8] - Close to box 1, far from box 2
        - Anchor 2: [0.7, 0.2] - Far from box 1, close to box 2
        - Anchor 3: [0.6, 0.6] - Medium distance from both boxes
        
        With pos_max_distance=0.5 and neg_min_distance=0.75:
        - Anchor 1: Positive match with box 1 (distance 0.3 < 0.5)
        - Anchor 2: Positive match with box 2 (distance 0.2 < 0.5)
        - Anchor 3: Ignore match (0.5 < distance 0.6 < 0.75)
    """
    # Input distances - 3 anchor points, 2 boxes
    distances = torch.tensor([
        [0.3, 0.8],  # Anchor 1: Close to box 1, far from box 2
        [0.7, 0.2],  # Anchor 2: Far from box 1, close to box 2
        [0.6, 0.6],  # Anchor 3: Medium distance from both boxes
    ])

    # Call function under test
    match_labels, match_idxs = _get_anchor_gt_matches(
        anchor_distances=distances,
        pos_max_distance=0.5,
        neg_min_distance=0.75,
    )

    # Expected outputs
    expected_labels = torch.tensor([1, 1, -1])  # Positive, Positive, Ignore
    expected_idxs = torch.tensor([0, 1, 0])     # Box 1, Box 2, Box 1 (closest)

    # Verify outputs
    assert torch.allclose(match_labels, expected_labels)
    assert torch.allclose(match_idxs, expected_idxs)


def test_get_bbox_offsets_values():
    """Test that bbox offsets are computed correctly.
    
    Example:
        Anchor points and GT bboxes:
        - Anchor 1: [0.5, 0.5] with GT box [0.2, 0.3, 0.9, 0.6]
          Expected offsets: [0.3, 0.2, 0.4, 0.1] * [image_height, image_width, image_height, image_width]
          (box extends different amounts in each direction)
        
        - Anchor 2: [0.2, 0.2] with GT box [0.1, 0.1, 0.3, 0.3]
          Expected offsets: [0.1, 0.1, 0.1, 0.1] * [image_height, image_width, image_height, image_width]
          (anchor is at center, box extends 0.1 in each direction)
        
        - Anchor 3: [0.8, 0.8] with GT box [0.7, 0.7, 0.9, 0.9]
          Expected offsets: [0.1, 0.1, 0.1, 0.1] * [image_height, image_width, image_height, image_width]
          (anchor is at center, box extends 0.1 in each direction)
    """
    # Input anchor points and GT bboxes
    anchor_points = torch.tensor([
        [0.5, 0.5],  # Center point
        [0.2, 0.2],  # Lower left point
        [0.8, 0.8],  # Upper right point
    ])
    
    gt_bboxes = torch.tensor([
        [0.2, 0.3, 0.9, 0.6],  # Box with different offsets in each direction
        [0.1, 0.1, 0.3, 0.3],  # Box centered at (0.2, 0.2)
        [0.7, 0.7, 0.9, 0.9],  # Box centered at (0.8, 0.8)
    ])

    # Image dimensions
    image_height = 224
    image_width = 224

    # Call function under test
    offsets = _get_bbox_offsets(
        anchor_points=anchor_points,
        gt_bboxes=gt_bboxes,
        image_height=image_height,
        image_width=image_width,
    )

    # Expected offsets (normalized values * image dimensions)
    expected_offsets = torch.tensor([
        [0.3, 0.2, 0.4, 0.1],  # Anchor 1: Box extends different amounts in each direction
        [0.1, 0.1, 0.1, 0.1],  # Anchor 2: Box extends 0.1 in each direction
        [0.1, 0.1, 0.1, 0.1],  # Anchor 3: Box extends 0.1 in each direction
    ]) * torch.tensor([image_height, image_width, image_height, image_width])

    # Verify outputs
    assert offsets.shape == (3, 4)
    assert torch.allclose(offsets, expected_offsets, atol=1e-6)

    # Verify that we can reconstruct the original boxes using the offsets
    reconstructed_boxes = torch.stack([
        anchor_points[..., 0] - offsets[..., 0] / image_height,  # min_row
        anchor_points[..., 1] - offsets[..., 1] / image_width,   # min_col
        anchor_points[..., 0] + offsets[..., 2] / image_height,  # max_row
        anchor_points[..., 1] + offsets[..., 3] / image_width,   # max_col
    ], dim=-1)
    
    assert torch.allclose(reconstructed_boxes, gt_bboxes, atol=1e-6)


def test_yad_loss_shapetype():
    """Test that the loss function outputs have correct shapes and types."""
    # Create model and test inputs
    num_classes = 80
    model = make_yad(num_classes=num_classes)
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)
    
    # Get model outputs
    bbox_out, cls_out, objectness_out = model(image)
    
    # Create ground truth bboxes
    gt_bboxes = [
        torch.tensor([
            [0.2, 0.3, 0.9, 0.6, 5],  # One box with class_id 5
            [0.1, 0.1, 0.3, 0.3, 10],  # Another box with class_id 10
        ]),
        torch.tensor([
            [0.4, 0.4, 0.6, 0.6, 15],  # One box with class_id 15
        ]),
    ]

    # Call loss function
    losses = model.loss((bbox_out, cls_out, objectness_out), gt_bboxes)

    # Check types
    assert isinstance(losses, dict)
    assert all(isinstance(v, torch.Tensor) for v in losses.values())
    assert all(v.dtype == torch.float32 for v in losses.values())

    # Check keys
    assert set(losses.keys()) == {"bbox_loss", "cls_loss", "objectness_loss"}

    # Check shapes - all losses should be scalar tensors
    assert all(v.ndim == 0 for v in losses.values())

    # Check values are non-negative
    assert all(v.item() >= 0 for v in losses.values()) 