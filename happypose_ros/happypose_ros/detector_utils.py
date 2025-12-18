import numpy as np
import torch
from torchvision.ops import nms
from happypose.toolbox.inference.types import DetectionsType, ObservationTensor


def create_crop_bboxes(h: int, w: int, scale=0.6):

    """
    Create 5 crop bounding boxes of equal sizes: center + 4 corners.

    :param h: image height
    :param w: image width
    :param scale: scale factor for crop size

    :return: crop_bboxes (N,4), bboxes (xyxy) to be used for crops
    """
    bh, bw = int(h * scale), int(w * scale)

    # central crop
    x1 = (w - bw) // 2
    y1 = (h - bh) // 2
    y2 = y1 + bh
    x2 = x1 + bw
    bbox_center = torch.tensor([x1,y1,x2,y2], dtype=torch.int32)

    # corner crops
    bbox_tl = torch.tensor([0,   0,   bw,bh], dtype=torch.int32)
    bbox_tr = torch.tensor([w-bw,0,   w, bh], dtype=torch.int32)
    bbox_bl = torch.tensor([0,   h-bh,bw,h ], dtype=torch.int32)
    bbox_br = torch.tensor([w-bw,h-bh,w ,h ], dtype=torch.int32)

    return torch.stack([bbox_center, bbox_tl, bbox_tr, bbox_bl, bbox_br])


def batch_im_crops(im_b: torch.Tensor, crop_bboxes: torch.Tensor):
    """
    im_b (B,C,H,W): batch of images 
    crop_bboxes (N,4): bboxes (xyxy) used to crop im_b

    return: im_b_crops: (B*N,C,H,W)
    """

    B, C, H, W = im_b.shape
    N = len(crop_bboxes)
    
    # Repeat the image B times along batch dimension
    # [im1, im1,..., im2, im2,..., im3, im3,...]
    im_b_repeated = im_b.repeat_interleave(N, dim=0)  # (B*N, C, H, W)
    x1, y1, x2, y2 = crop_bboxes.T
    # Crop all boxes at once
    crops = torch.stack([
        im_b_repeated[N*i+j, :, y1[j]:y2[j], x1[j]:x2[j]] 
        for i in range(B)
        for j in range(N)
    ])  # (N*B, C, crop_h, crop_w)

    return crops



def aggregate_crop_detections(detections: DetectionsType, crop_bboxes: torch.Tensor, image_batch_size: int):
    """
    Transform local crop detections to global image detections and aggregate them.
    """

    df = detections.infos
    N = crop_bboxes.shape[0]
    first_img_ids = N*np.arange(image_batch_size)

    # transform detections from local to global
    for i, crop_bb in enumerate(crop_bboxes):
        batch_ids_crop_i = first_img_ids + i
        row_ids = df.index[df["batch_im_id"].isin(batch_ids_crop_i)].to_numpy()
        if row_ids.size == 0:
            continue
        row_ids_t = torch.as_tensor(row_ids, device=crop_bboxes.device)
        # local to global: xg, yg = xl + x1, yl + y1  
        detections.bboxes[row_ids_t, 0] += crop_bb[0]
        detections.bboxes[row_ids_t, 2] += crop_bb[0]
        detections.bboxes[row_ids_t, 1] += crop_bb[1]
        detections.bboxes[row_ids_t, 3] += crop_bb[1]

    # aggregate global detections -> simply regroup the contiguous batch ids
    df["batch_im_id"] = df["batch_im_id"] // N
    detections.df = df

    return detections


def nms_batches(detections: DetectionsType, iou_threshold=0.8):
    """Apply Non-Maximum Suppression within each batch."""
    if len(detections.infos) == 0:
        return detections
    df = detections.infos
    groups = df.groupby(["batch_im_id"], group_keys=False)
    all_scores = torch.tensor(df.score).to(detections.bboxes.device)
    all_nms_ids = []
    for _, g in groups:
        ts_ids = g.index.to_numpy()
        ids_nms = nms(detections.bboxes[ts_ids], all_scores[ts_ids], iou_threshold)
        all_nms_ids += ts_ids[ids_nms.cpu().numpy()].tolist()
    
    return detections[np.asarray(all_nms_ids)]


def best_of_batches(detections):
    """"
    Keep the detection with the best score out of the 

    """
    groups = detections.infos.groupby(["batch_im_id"], group_keys=False)
    # Get indices of maximum scores for each group
    best_indices = groups.apply(lambda x: x.score.idxmax(), include_groups=False).values
    detections = detections[best_indices]
    return detections


def get_multicrop_detections(detector, images: torch.Tensor, K: torch.Tensor, device, tile_detection_scale: float = 0.6, detector_args: dict = {}):
    """ 
    Get detections using multi-crop strategy.
    TODO: detect objects in the whole image as well and aggregate results.

    detector: happypose detector model
    images: (B,C,H,W) input images
    K: (B,3,3) camera intrinsics
    device: torch device
    tile_detection_scale: scale factor for crop size 
    detector_args: additional arguments for detector
    """

    # for multi crop detection
    h, w = images.shape[2:4]
    crop_bboxes = create_crop_bboxes(h, w, tile_detection_scale)
    images_crops = batch_im_crops(images, crop_bboxes)
    obs_crop = ObservationTensor(images_crops, K).to(device)
    detections = detector.get_detections(obs_crop, **detector_args)
    detections = aggregate_crop_detections(detections, crop_bboxes, images.shape[0])
    detections = nms_batches(detections)
    return detections