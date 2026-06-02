#!/usr/bin/env python3
"""Visualize ROI features: cropped image regions from query predictions."""
import sys, os, torch, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, '/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching')
from detectron2.config import get_cfg
from configs.defaults import add_dataset_config, add_scenegraph_config
from detectron2.structures import ImageList
import torch.nn.functional as F

CKPT = 'z_outputs/overfit_resnet_roi_bs12_16000/model_0015999.pth'
OUT_DIR = 'z_outputs/overfit_resnet_roi_bs12_16000/debug_roi'
os.makedirs(OUT_DIR, exist_ok=True)

cfg = get_cfg()
add_dataset_config(cfg); add_scenegraph_config(cfg)
cfg.merge_from_file('configs/speaq_actiongenome_minimal.yaml')
cfg.MODEL.SAM3.ENABLED = False; cfg.MODEL.TEMPORAL.ENABLED = False
cfg.MODEL.ROI_REFINE.ENABLED = True; cfg.MODEL.ROI_REFINE.LOSS_ENABLED = True
cfg.MODEL.ROI_REFINE.RESNET_FPN_LEVEL = 1; cfg.MODEL.ROI_REFINE.STRIDE = 16
cfg.MODEL.ROI_REFINE.APPLY_TO = 'all'; cfg.MODEL.ROI_REFINE.SMALL_AREA_THRESH = 0.02
cfg.MODEL.DEVICE = 'cuda'
cfg.DATASETS.ACTION_GENOME.ANNOTATIONS = 'dataset_overfit_temporal/annotations'
cfg.DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN = -1
cfg.DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL = 0
cfg.DATASETS.ACTION_GENOME.FRAMES = 'dataset/frames'

from SpeaQ.data.tools.utils import register_datasets
register_datasets(cfg)
from detectron2.data import DatasetCatalog, MetadataCatalog
import SpeaQ.modeling.meta_arch.detr
from detectron2.modeling import build_model

model = build_model(cfg)
ckpt = torch.load(CKPT, map_location='cpu')
state = ckpt.get('model', ckpt)
md = model.state_dict()
f = {k: v for k, v in state.items() if k in md and md[k].shape == v.shape}
model.load_state_dict(f, strict=False)
model.to('cuda').eval()
print(f'Loaded {len(f)} keys')

dataset_dicts = DatasetCatalog.get('AG_train')
thing_classes = MetadataCatalog.get('AG_train').thing_classes

for idx in [0, 3, 5]:
    d = dataset_dicts[idx]
    img = Image.open(d['file_name']).convert('RGB')
    iw, ih = img.size
    img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float().unsqueeze(0).cuda()
    images = ImageList(img_tensor, [(ih, iw)])
    
    with torch.no_grad():
        # Monkey-patch to capture ROI features
        orig_fwd = model.detr.roi_refine_head.forward
        
        captured = {}
        def hook_fwd(self, embeddings, boxes_cxcywh, feature, image_h, image_w, labels=None):
            captured['boxes'] = boxes_cxcywh.clone()
            captured['feature'] = feature.clone()
            ret = orig_fwd(self, embeddings, boxes_cxcywh, feature, image_h, image_w, labels)
            captured['gate'] = getattr(self, '_last_gate_stats', None)
            return ret
        model.detr.roi_refine_head.forward = hook_fwd.__get__(model.detr.roi_refine_head)
        
        outputs = model(images)
    
    # Get top subject predictions
    logits_s = outputs.get('pred_logits_subject', None)
    boxes_s = outputs.get('pred_boxes_subject', None)
    if logits_s is None: continue
    
    probs = F.softmax(logits_s[0], dim=-1)
    scores, labels = probs[..., :-1].max(dim=-1)
    
    # Draw all predicted boxes on image
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    for qi in range(min(20, scores.shape[0])):
        if scores[qi] < 0.3: continue
        cx, cy, w, h = boxes_s[0, qi].cpu().tolist()
        x1 = int(max(0, (cx - w/2) * iw))
        y1 = int(max(0, (cy - h/2) * ih))
        x2 = int(min(iw, (cx + w/2) * iw))
        y2 = int(min(ih, (cy + h/2) * ih))
        cls_name = thing_classes[labels[qi].item()]
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1, max(0, y1-10)), f'{cls_name} {scores[qi]:.2f}', fill='red')
    draw_img.save(os.path.join(OUT_DIR, f'frame_{idx}_boxes.jpg'))
    
    # Save top-5 crops
    topk = min(5, scores.shape[0])
    for k in range(topk):
        cx, cy, w, h = boxes_s[0, k].cpu().tolist()
        x1 = int(max(0, (cx - w/2) * iw))
        y1 = int(max(0, (cy - h/2) * ih))
        x2 = int(min(iw, (cx + w/2) * iw))
        y2 = int(min(ih, (cy + h/2) * ih))
        if x2 > x1 and y2 > y1:
            crop = img.crop((x1, y1, x2, y2))
            cls_name = thing_classes[labels[k].item()]
            crop.save(os.path.join(OUT_DIR, f'frame_{idx}_crop{k}_{cls_name}_{scores[k]:.2f}.jpg'))
    
    gate = captured.get('gate', {})
    if gate:
        print(f'Frame {idx}: gate_mean={gate.get("mean","?"):.3f}, small_gate={gate.get("small_gate_mean","?"):.3f}, large_gate={gate.get("large_gate_mean","?"):.3f}')

print(f'Output in {OUT_DIR}')
