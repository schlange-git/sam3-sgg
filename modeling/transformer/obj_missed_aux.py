import torch
import torch.nn.functional as F
from .util.utils import box_cxcywh_to_xyxy


def box_iou_xyxy(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def build_quality_aware_aux_indices(outputs, targets, combined_indices, cfg):
    """Find unmatched high-quality queries to ignore in primary CE.

    A candidate must be unmatched by the main relation matcher, predict the same
    class as at least one GT object, and have IoU above the branch threshold.
    """
    device = outputs["relation_subject_logits"].device
    B, N, _ = outputs["relation_subject_logits"].shape

    subject_indices = []
    object_indices = []
    num_sub = 0
    num_obj = 0
    iou_thresh = float(getattr(cfg, "IOU_THRESH", 0.75))
    iou_thresh_sub = float(getattr(cfg, "IOU_THRESH_SUB", iou_thresh))
    iou_thresh_obj = float(getattr(cfg, "IOU_THRESH_OBJ", iou_thresh))
    min_score = float(getattr(cfg, "MIN_SCORE", 0.0))
    min_score_sub = float(getattr(cfg, "MIN_SCORE_SUB", min_score))
    min_score_obj = float(getattr(cfg, "MIN_SCORE_OBJ", min_score))
    apply_sub = bool(getattr(cfg, "APPLY_SUBJECT", True))
    apply_obj = bool(getattr(cfg, "APPLY_OBJECT", True))
    do_log = bool(getattr(cfg, "DEBUG", False))

    stats = {
        "sub_bg": 0,
        "sub_score": 0,
        "sub_cls": 0,
        "obj_bg": 0,
        "obj_score": 0,
        "obj_cls": 0,
    }
    all_sub_ious = []
    all_sub_scores = []
    all_obj_ious = []
    all_obj_scores = []
    all_sub_candidate_ious = []
    all_obj_candidate_ious = []

    for b in range(B):
        gt_boxes = targets[b]["combined_boxes"]
        gt_labels = targets[b]["combined_labels"]
        G = gt_boxes.shape[0]
        empty = torch.empty(0, dtype=torch.long, device=device)

        if G == 0:
            subject_indices.append((empty, empty))
            object_indices.append((empty, empty))
            continue

        sub_src, _ = combined_indices["subject"][b]
        obj_src, _ = combined_indices["object"][b]
        all_q = torch.arange(N, dtype=torch.long, device=device)

        # --- Subject branch ---
        if apply_sub:
            sub_used = torch.zeros(N, dtype=torch.bool, device=device)
            if sub_src.numel() > 0:
                sub_used[sub_src] = True
            sub_bg = all_q[~sub_used]

            sub_logits = outputs["relation_subject_logits"][b]
            sub_boxes = outputs["relation_subject_boxes"][b]
            sub_prob = F.softmax(sub_logits, dim=-1)
            sub_score = sub_prob[:, :-1].max(dim=-1).values
            sub_cls = sub_prob[:, :-1].argmax(dim=-1)
            sub_xyxy = box_cxcywh_to_xyxy(sub_boxes)
            gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)

            mq, mt = [], []
            stats["sub_bg"] += int(sub_bg.numel())
            for qi in sub_bg.tolist():
                if sub_score[qi].item() < min_score_sub:
                    continue
                stats["sub_score"] += 1
                qc = int(sub_cls[qi])
                cm = (gt_labels == qc).nonzero(as_tuple=False).squeeze(-1)
                if cm.numel() == 0:
                    continue
                stats["sub_cls"] += 1
                ious = box_iou_xyxy(sub_xyxy[qi:qi+1], gt_xyxy[cm])[0]
                bi, bk = torch.max(ious, dim=0)
                if do_log:
                    all_sub_candidate_ious.append(float(bi.item()))
                if float(bi.item()) >= iou_thresh_sub:
                    mq.append(qi)
                    mt.append(int(cm[bk].item()))
                    if do_log:
                        all_sub_ious.append(float(bi.item()))
                        all_sub_scores.append(float(sub_score[qi].item()))
            if mq:
                subject_indices.append((
                    torch.as_tensor(mq, dtype=torch.long, device=device),
                    torch.as_tensor(mt, dtype=torch.long, device=device)))
                num_sub += len(mq)
            else:
                subject_indices.append((empty, empty))
        else:
            subject_indices.append((empty, empty))
            if do_log:
                import logging
                logger = logging.getLogger("detectron2")
                print("[QualityAux DEBUG] do_log=%s sub_bg=%d sub_score=%d sub_cand=%d sub_match=%d" % (do_log, stats["sub_bg"], stats["sub_score"], len(all_sub_candidate_ious), len(all_sub_ious)), flush=True)

            # Sub diagnostic: candidate IoU/score distribution
            if do_log and len(all_sub_candidate_ious) > 0:
                sc = torch.tensor(all_sub_candidate_ious)
                logger.info("[QualityAux] sub candidate IoU: min=%.3f mean=%.3f max=%.3f (n=%d)" % (sc.min().item(), sc.mean().item(), sc.max().item(), len(all_sub_candidate_ious)))
            if do_log and len(all_sub_scores) > 0:
                ss = torch.tensor(all_sub_scores)
                logger.info("[QualityAux] sub matched score: min=%.3f mean=%.3f max=%.3f (n=%d)" % (ss.min().item(), ss.mean().item(), ss.max().item(), len(all_sub_scores)))
            if do_log and len(all_sub_scores) == 0 and stats["sub_score"] > 0:
                logger.info("[QualityAux] sub: %d passed score filter but 0 matched IoU (thresh=%.2f)" % (stats["sub_score"], iou_thresh_sub))
            if do_log and stats["sub_score"] == 0 and stats["sub_bg"] > 0:
                logger.info("[QualityAux] sub: 0/%d passed score filter (thresh=%.2f)" % (stats["sub_bg"], min_score_sub))

        # --- Object branch ---
        if apply_obj:
            obj_used = torch.zeros(N, dtype=torch.bool, device=device)
            if obj_src.numel() > 0:
                obj_used[obj_src] = True
            obj_bg = all_q[~obj_used]

            obj_logits = outputs["relation_object_logits"][b]
            obj_boxes = outputs["relation_object_boxes"][b]
            obj_prob = F.softmax(obj_logits, dim=-1)
            obj_score = obj_prob[:, :-1].max(dim=-1).values
            obj_cls = obj_prob[:, :-1].argmax(dim=-1)
            obj_xyxy = box_cxcywh_to_xyxy(obj_boxes)
            gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)

            mq, mt = [], []
            stats["obj_bg"] += int(obj_bg.numel())
            for qi in obj_bg.tolist():
                if obj_score[qi].item() < min_score_obj:
                    continue
                stats["obj_score"] += 1
                qc = int(obj_cls[qi])
                cm = (gt_labels == qc).nonzero(as_tuple=False).squeeze(-1)
                if cm.numel() == 0:
                    continue
                stats["obj_cls"] += 1
                ious = box_iou_xyxy(obj_xyxy[qi:qi+1], gt_xyxy[cm])[0]
                bi, bk = torch.max(ious, dim=0)
                if do_log:
                    all_obj_candidate_ious.append(float(bi.item()))
                if float(bi.item()) >= iou_thresh_obj:
                    mq.append(qi)
                    mt.append(int(cm[bk].item()))
                    if do_log:
                        all_obj_ious.append(float(bi.item()))
                        all_obj_scores.append(float(obj_score[qi].item()))
            if mq:
                object_indices.append((
                    torch.as_tensor(mq, dtype=torch.long, device=device),
                    torch.as_tensor(mt, dtype=torch.long, device=device)))
                num_obj += len(mq)
            else:
                object_indices.append((empty, empty))
        else:
            object_indices.append((empty, empty))

    # per-batch stats returned to criterion for aggregated logging
    return {
        "subject": subject_indices,
        "object": object_indices,
        "_cum_sub": num_sub,
        "_cum_obj": num_obj,
        "_sub_ious": all_sub_ious,
        "_sub_scores": all_sub_scores,
        "_sub_cand_ious": all_sub_candidate_ious,
        "_obj_ious": all_obj_ious,
        "_obj_scores": all_obj_scores,
        "_obj_cand_ious": all_obj_candidate_ious,
    }
