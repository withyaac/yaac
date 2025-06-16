"""Yet Another Detector"""

import torch
from typing import Dict, Tuple, List
import timm


class YAD(torch.nn.Module):
    """Yet Another Detector


    img --> backbone --> neck   --> bbox_head --------> bbox_out
                                --> cls_head  --------> cls_out
                                --> objectness_head --> objectness_out

    Args:
        backbone: Backbone network
        neck: Neck network
        bbox_head: Bbox head network
        cls_head: Classification head network
        objectness_head: Objectness head network
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        neck: torch.nn.Module,
        bbox_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        objectness_head: torch.nn.Module,
    ):
        super().__init__()
        self._backbone = backbone
        self._neck = neck
        self._bbox_head = bbox_head
        self._cls_head = cls_head
        self._objectness_head = objectness_head

        # Anchor points depend on the feature map size, so we'll keep track
        # of the latest feature map height/width so we can cahce the anchors 
        self._anchor_points = None
        self._feature_map_height = -1
        self._feature_map_width = -1
        self._image_height = -1
        self._image_width = -1

    def forward(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            img: Input image
        Returns:
            bbox_out: Bbox output
            cls_out: Classification output
            objectness_out: Objectness output
        """
        backbone_features = self._backbone(img)
        neck_features = self._neck(backbone_features)
        bbox_out = self._bbox_head(neck_features)
        cls_out = self._cls_head(neck_features)
        objectness_out = self._objectness_head(neck_features)

        # update anchors cache
        feature_map_height = backbone_features.shape[2]
        feature_map_width = backbone_features.shape[3]
        self._image_height = img.shape[2]
        self._image_width = img.shape[3]
        if self._anchor_points is None or feature_map_height != self._feature_map_height or feature_map_width != self._feature_map_width:
            self._feature_map_height = feature_map_height
            self._feature_map_width = feature_map_width
            self._anchor_points = _make_anchor_points(
                feature_map_height=feature_map_height,
                feature_map_width=feature_map_width

            )

        return bbox_out, cls_out, objectness_out

    def loss(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        gt_bboxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Loss function

        Args:
            outputs: Outputs from forward pass
            gt_bboxes: Ground truth bboxes. Length is the batch size. Each element 
                is a tensor of shape (N, 5) where N is the number of bboxes. The 5
                values are (min_row, min_col, max_row, max_col, class_id). The boox 
                coordinates are normalized to [0,1], in the coordinate system of the
                inputs to the forward method.
        Returns:
            Keys are loss names, values are loss values (scalar tensors)
        """
        batch_size = len(gt_bboxes)
        objectness_loss_per_sample = []
        cls_loss_per_sample = []
        bbox_loss_per_sample = []
        breakpoint()
        for batch_idx in range(batch_size):
            anchor_distances = _get_pairwise_distance_points_bboxes(
                anchor_points=self._anchor_points,
                gt_bboxes=gt_bboxes[batch_idx]
            )
            match_labels, match_idxs = _get_anchor_gt_matches(
                anchor_distances,
            )

            # Objectness loss
            #   For each anchor, objectness is a binary classification equal to the 
            #   match label, from the logit output from the objectness head.
            objectness_logits = outputs[2][batch_idx].reshape(-1)
            objectness_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                objectness_logits,
                match_labels.float(),
            )
            objectness_loss_per_sample.append(objectness_loss)
            pos_mask = match_labels == 1

            # Classification Loss
            #   For each anchor that got a positive match label, the classification
            #     loss is cross entropy of the classiication logits
            num_classes = outputs[1].shape[1]
            cls_logits = outputs[1][batch_idx].reshape(num_classes,-1).permute(1,0)
            pos_cls_logits = cls_logits[pos_mask]
            pos_match_idxs = match_idxs[pos_mask]
            gt_cls_ids = gt_bboxes[batch_idx][pos_match_idxs, 4]
            cls_loss = torch.nn.functional.cross_entropy(
                pos_cls_logits,
                gt_cls_ids.long(),
            )
            cls_loss_per_sample.append(cls_loss)


            ## BBox loss
            #    For each anchor that got a positive match label, 
            #     loss is smooth l1 loss of the bbox offsets
            pos_anchor_points = self._anchor_points[pos_mask]
            pos_gt_bboxes = gt_bboxes[batch_idx][pos_match_idxs, :4]
            pos_bbox_offsets = _get_bbox_offsets(
                anchor_points=pos_anchor_points,
                gt_bboxes=pos_gt_bboxes,
                image_height=self._image_height,
                image_width=self._image_width,
            )
            # convert to image coordinates
            pred_bbox_offsets = outputs[0][batch_idx]
            pred_bbox_offsets = pred_bbox_offsets.reshape(4, -1).permute(1,0)
            pos_pred_bbox_offsets = pred_bbox_offsets[pos_mask]
            bbox_loss = torch.nn.functional.smooth_l1_loss(
                pos_pred_bbox_offsets,
                pos_bbox_offsets,
            )
            bbox_loss_per_sample.append(bbox_loss)


        objectness_loss = torch.tensor(objectness_loss_per_sample).mean()
        cls_loss = torch.tensor(cls_loss_per_sample).mean()
        bbox_loss = torch.tensor(bbox_loss_per_sample).mean()

        return {
            "bbox_loss": bbox_loss,
            "cls_loss": cls_loss,
            "objectness_loss": objectness_loss,
        }

    def postprocess(
        self, outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Postprocess outputs

        Args:
            outputs: Outputs from forward pass

        Returns:
            The detected bboxes. Shape (B, N, 6).
            - B is the batch size.
            - N is the number of detected bboxes.
            - 6 is the bounding box format: (min_row, min_col, max_row, max_col, score, class_id).
        """
        pass

def _make_anchor_points(
    feature_map_height: int,
    feature_map_width: int,
) -> torch.Tensor:
    """Make anchor points, a grid of points on the feature map.

    Each pixel in the feature map is an anchor point. The points are placed in a grid
    and normalized to [0,1] coordinates.

    Args:
        feature_map_height: Height of the feature map.
        feature_map_width: Width of the feature map.

    Returns:
        Anchor points in normalized coordinates [0,1]. Shape (num_points, 2).
        num_points = feature_map_height * feature_map_width
    """
    # Create a grid of points on the feature map, normalized to [0,1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, feature_map_height),
        torch.linspace(0, 1, feature_map_width),
        indexing='ij'  # Use 'ij' indexing to match the test's expected order
    )

    # Flatten the grid and stack into points
    anchor_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    return anchor_points

def _get_pairwise_distance_points_bboxes(
    anchor_points: torch.Tensor,
    gt_bboxes: torch.Tensor, # TODO: rename this arg to just "bboxes"
) -> torch.Tensor:
    """Get the distance between all pairs of anchor points and ground truth bboxes.

    The distance between a point and a box is as follows:
    - If the point is outside the box, the distance is 1.
    - If the point is inside the box, the distance is the distance from the point to
        center of the box, normalized to [0,1], so that a point in a corner of the box
        has distance 1.

    Args:
        anchor_points: Anchor points. Shape (num_points, 2). 
        In normalized coordinates [0,1]: (row, col).
        gt_bboxes: Ground truth bboxes. Shape (num_bboxes, 4).
        In normalized coordinates [0,1]: (min_row, min_col, max_row, max_col).
    Returns:
        Distance between all pairs of anchor points and ground truth bboxes.
        Shape (num_points, num_bboxes).
    """
    num_points = anchor_points.shape[0]
    num_bboxes = gt_bboxes.shape[0]

    points = anchor_points.unsqueeze(1)  # (num_points, 1, 2)
    boxes = gt_bboxes.unsqueeze(0)  # (1, num_bboxes, 4)

    point_rows = points[..., 0]
    point_cols = points[..., 1]
    box_min_rows = boxes[..., 0]
    box_min_cols = boxes[..., 1]
    box_max_rows = boxes[..., 2]
    box_max_cols = boxes[..., 3]

    inside_row = (point_rows >= box_min_rows) & (point_rows <= box_max_rows)
    inside_col = (point_cols >= box_min_cols) & (point_cols <= box_max_cols)
    inside = inside_row & inside_col  # (num_points, num_bboxes)

    box_center_rows = (box_min_rows + box_max_rows) / 2.0
    box_center_cols = (box_min_cols + box_max_cols) / 2.0

    box_d_rows = (box_max_rows - box_min_rows) / 2.0
    box_d_cols = (box_max_cols - box_min_cols) / 2.0
    box_max_dist = torch.sqrt(box_d_rows ** 2 + box_d_cols ** 2)  # (1, num_bboxes)

    # Compute distance from point to center
    dist_to_center = torch.sqrt(
        (point_rows - box_center_rows) ** 2 + (point_cols - box_center_cols) ** 2
    )  # (num_points, num_bboxes)

    # Avoid division by zero for degenerate boxes
    box_max_dist = torch.clamp(box_max_dist, min=1e-8)
    norm_dist = dist_to_center / box_max_dist  # (num_points, num_bboxes)

    # If point is inside box, use normalized distance; else, distance is 1
    distance = torch.where(inside, norm_dist, torch.ones_like(norm_dist))

    return distance
    
def _get_anchor_gt_matches(
    anchor_distances: torch.Tensor,
    pos_max_distance: float = 0.5,
    neg_min_distance: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match anchors to gt bboxes based on distances.

    An anchor can match to at most 1 GT bbox. Multple anchors can match to the same
        gt bbox. Each anchor will be matched to its closest gt bbox.
    
    Args:
        anchor_distances: the normalized distance [0,1] from anchor to bbox.
        pos_max_distance: maximum distance a point can have from a bbox and still
            be considered a positive match.
        neg_max_distance: minium distance a point can have from a bbox and be considered
            a negative match.


    Returns:
        A tuple with:
            - match labels: Whether the anchor matched with a GT bbox. 
                Shape (num_points,). Values: 0: no match, 1: positive match, 
                -1: "ignore" (distance is between pos_max_distance and neg_min_distance)
            - match_idxs: The index of the bbox the point matches. Shape (num_points,).
                Dtype: torch.long. Every anchor will have a match idx, even
                if it does not have  match_label of 1.
    """
    num_points, num_bboxes = anchor_distances.shape
    
    # For each anchor point, find the closest bbox
    min_distances, match_idxs = torch.min(anchor_distances, dim=1)
    
    # Initialize match labels
    match_labels = torch.zeros(num_points, dtype=torch.long, device=anchor_distances.device)
    
    # Set positive matches (distance <= pos_max_distance)
    pos_mask = min_distances <= pos_max_distance
    match_labels[pos_mask] = 1
    
    # Set negative matches (distance >= neg_min_distance)
    neg_mask = min_distances >= neg_min_distance
    match_labels[neg_mask] = 0
    
    # Set ignore matches (pos_max_distance < distance < neg_min_distance)
    ignore_mask = ~(pos_mask | neg_mask)
    match_labels[ignore_mask] = -1
    
    return match_labels, match_idxs

def _get_bbox_offsets(
    anchor_points: torch.Tensor,
    gt_bboxes: torch.Tensor, 
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Get the offsets from anchor points to ground truth bboxes.
    
    The offsets are what is predicted by the bbox head. They are the relative
    position of the anchor point to the ground truth bbox.
    
    The input to this function is N anchor points and N ground truth bboxes.
    The output is N bbox offsets. It is assumed that the anchor points and ground
    truth bboxes are aligned, i.e. the first anchor point and first ground truth
    bbox are "matched" to each other.

    Args:
        anchor_points: Anchor points. Shape (N, 2).
        gt_bboxes: Ground truth bboxes. Shape (N, 4).
        image_height: Height of the image. To convert the offsets to absolute
            coordinates.
        image_width: Width of the image. To convert the offsets to absolute
            coordinates.
        
    Returns:
        Bbox offsets, in abosulte image coordinates (pixels). Shape (N, 4). 
        The 4 values are:
            - offset of the min row of the bbox relative to the anchor point
            - offset of the min col of the bbox relative to the anchor point
            - offset of the max row of the bbox relative to the anchor point
            - offset of the max col of the bbox relative to the anchor point

        So, to reconstruct the bbox, we would do:
            bbox_min_row = anchor_point_row - bbox_offset[0]
            bbox_min_col = anchor_point_col - bbox_offset[1]
            bbox_max_row = anchor_point_row + bbox_offset[2]
            bbox_max_col = anchor_point_col + bbox_offset[3]  
    """
    anchor_rows = anchor_points[..., 0]
    anchor_cols = anchor_points[..., 1]
    gt_bboxes_min_rows = gt_bboxes[..., 0]
    gt_bboxes_min_cols = gt_bboxes[..., 1]
    gt_bboxes_max_rows = gt_bboxes[..., 2]
    gt_bboxes_max_cols = gt_bboxes[..., 3]
    
    bbox_offsets = torch.stack([
        anchor_rows - gt_bboxes_min_rows,
        anchor_cols - gt_bboxes_min_cols,
        gt_bboxes_max_rows - anchor_rows,
        gt_bboxes_max_cols - anchor_cols,
    ], dim=-1)
    bbox_offsets = bbox_offsets * torch.tensor([image_height, image_width, image_height, image_width], device=bbox_offsets.device)
    return bbox_offsets

def make_yad(num_classes: int = 80) -> YAD:
    """Make a YAD model.

    This function creates a YAD model with the following components:

    Args:
        num_classes: Number of classes for classification head. Defaults to 80 (COCO).

    Returns:
        A YAD model.
    """
    backbone = timm.create_model("resnet18", pretrained=True)
    input_normalization_kwargs = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    # Break up the backbone into a list of modules, then wrap in a Sequential
    if hasattr(backbone, "children"):
        backbone_modules = list(backbone.children())
        chosen_modules = backbone_modules[:-4]
        backbone = torch.nn.Sequential(*chosen_modules)
    else:
        raise ValueError("Backbone must have children")

    backbone_out_channels = 128
    
    # The typical order is: Conv -> BatchNorm -> Activation
    neck = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=backbone_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(backbone_out_channels),
        torch.nn.ReLU(inplace=True),
    )

    # BBox head: two convs
    bbox_head = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=backbone_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(backbone_out_channels),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        ),
    )

    # Objectness head: two convs, one output channel
    objectness_head = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=backbone_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(backbone_out_channels),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=1,  # One channel for objectness
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        ),
    )

    # Classification head: two convs, num_classes output channels
    cls_head = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=backbone_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(backbone_out_channels),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=num_classes,  # One channel per class
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        ),
    )

    return YAD(
        backbone=backbone,
        neck=neck,
        bbox_head=bbox_head,
        cls_head=cls_head,
        objectness_head=objectness_head,
    )





if __name__ == "__main__":
    model = make_yad()
    image = torch.randn(1, 3, 224, 224)
    outputs = model(image)
    breakpoint()
    abc = 1
    
    