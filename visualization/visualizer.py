"""
Scene Graph Visualizer for visualizing scene graph predictions
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List, Tuple


class SceneGraphVisualizer:
    """Visualizer for scene graph predictions"""
    
    def __init__(self, metadata, output_dir: str):
        """
        Args:
            metadata: MetadataCatalog object containing class names
            output_dir: Directory to save visualizations
        """
        self.metadata = metadata
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Get class names
        self.thing_classes = metadata.thing_classes if hasattr(metadata, 'thing_classes') else []
        self.predicate_classes = metadata.predicate_classes if hasattr(metadata, 'predicate_classes') else []
        
    def visualize_scene_graph(
        self,
        image_path: str,
        boxes: np.ndarray,
        labels: np.ndarray,
        relations: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        rel_scores: Optional[np.ndarray] = None,
        output_name: Optional[str] = None,
        top_k_relations: int = 100,
        score_threshold: float = 0.3,
    ):
        """
        Visualize scene graph predictions
        
        Args:
            image_path: Path to input image
            boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
            labels: Object class labels [N]
            relations: Relations [M, 3] in (subject_idx, object_idx, predicate) format
            scores: Object scores [N] (optional)
            rel_scores: Relation scores [M] (optional)
            output_name: Output filename (without extension)
            top_k_relations: Number of top relations to show
            score_threshold: Minimum score threshold for relations
        """
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size
        
        # Filter relations by score if provided
        if relations is not None and rel_scores is not None and len(relations) > 0:
            # Filter by threshold
            if len(rel_scores) == len(relations):
                keep = np.array(rel_scores) >= score_threshold
                relations = relations[keep]
                rel_scores = np.array(rel_scores)[keep] if isinstance(rel_scores, np.ndarray) else [rel_scores[i] for i in range(len(rel_scores)) if keep[i]]
            
            # Take top k
            if len(relations) > top_k_relations:
                if rel_scores is not None and len(rel_scores) == len(relations):
                    top_indices = np.argsort(rel_scores)[-top_k_relations:][::-1]
                    relations = relations[top_indices]
                    rel_scores = np.array(rel_scores)[top_indices] if isinstance(rel_scores, np.ndarray) else [rel_scores[i] for i in top_indices]
                else:
                    relations = relations[:top_k_relations]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        ax.axis('off')
        
        # Draw boxes and labels
        colors = plt.cm.get_cmap('tab20', len(self.thing_classes) if self.thing_classes else 20)
        def _format_score(v):
            # Avoid visually misleading "0.00" for tiny but non-zero confidences.
            if v is None:
                return ""
            return f"{float(v):.4f}" if float(v) < 0.1 else f"{float(v):.2f}"

        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            # Clip to image boundaries
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            
            width = x2 - x1
            height = y2 - y1
            
            if width > 0 and height > 0:
                # Draw box
                color = colors(label % colors.N)
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Draw label
                label_text = self.thing_classes[label] if label < len(self.thing_classes) else f"Class {label}"
                if scores is not None and i < len(scores):
                    label_text += f" {_format_score(scores[i])}"
                
                ax.text(
                    x1, y1 - 5, label_text,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    fontsize=10, color='black'
                )
        
        # Draw relations
        if relations is not None and len(relations) > 0:
            used_text_positions: List[Tuple[float, float]] = []
            def _avoid_overlap(x, y):
                # Small deterministic offsets when relation texts overlap.
                candidate_offsets = [
                    (0, 0), (0, -14), (0, 14), (14, 0), (-14, 0),
                    (14, -14), (-14, -14), (14, 14), (-14, 14),
                    (28, 0), (-28, 0),
                ]
                for dx, dy in candidate_offsets:
                    cx, cy = x + dx, y + dy
                    collision = False
                    for ux, uy in used_text_positions:
                        if abs(cx - ux) < 38 and abs(cy - uy) < 14:
                            collision = True
                            break
                    if not collision:
                        used_text_positions.append((cx, cy))
                        return cx, cy
                used_text_positions.append((x, y))
                return x, y

            for rel in relations:
                if len(rel) >= 3:
                    sub_idx, obj_idx, pred_idx = int(rel[0]), int(rel[1]), int(rel[2])
                    if sub_idx < len(boxes) and obj_idx < len(boxes):
                        sub_box = boxes[sub_idx]
                        obj_box = boxes[obj_idx]
                        
                        # Get center points
                        sub_center = ((sub_box[0] + sub_box[2]) / 2, (sub_box[1] + sub_box[3]) / 2)
                        obj_center = ((obj_box[0] + obj_box[2]) / 2, (obj_box[1] + obj_box[3]) / 2)
                        
                        # Draw arrow
                        ax.annotate(
                            '', xy=obj_center, xytext=sub_center,
                            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.6)
                        )
                        
                        # Draw predicate label at midpoint
                        pred_text = self.predicate_classes[pred_idx] if pred_idx < len(self.predicate_classes) else f"Pred {pred_idx}"
                        if rel_scores is not None and len(rel_scores) >= len(relations):
                            # Find matching relation score
                            for j, r in enumerate(relations):
                                if j < len(rel_scores) and r[0] == sub_idx and r[1] == obj_idx:
                                    pred_text += f" {_format_score(rel_scores[j])}"
                                    break
                        
                        mid_x = (sub_center[0] + obj_center[0]) / 2
                        mid_y = (sub_center[1] + obj_center[1]) / 2
                        mid_x, mid_y = _avoid_overlap(mid_x, mid_y)
                        ax.text(
                            mid_x, mid_y, pred_text,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                            fontsize=9, ha='center', va='center'
                        )
        
        # Save figure
        if output_name:
            save_path = os.path.join(self.output_dir, f"{output_name}.png")
        else:
            basename = os.path.basename(image_path)
            save_path = os.path.join(self.output_dir, f"vis_{os.path.splitext(basename)[0]}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path

