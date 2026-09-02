"""
Manufacturing Inspection Dataset Generator for Edge Anomaly Detection.
Generates synthetic manufacturing parts with normal textures and controlled defect patterns
(cracks, scratches, voids, stains) for visual quality control.
"""

import os
import random
import json
from pathlib import Path
import numpy as np
import cv2


def generate_base_metal_surface(size: int = 128) -> np.ndarray:
    """Generate a realistic brushed metal surface with textured background."""
    # Base gray metal tone with subtle gradient
    base_val = random.randint(160, 190)
    img = np.full((size, size), base_val, dtype=np.float32)
    
    # Add horizontal/vertical brushed texture lines
    lines_intensity = np.random.normal(0, 4, (size, size)).astype(np.float32)
    img += cv2.GaussianBlur(lines_intensity, (5, 1), 0)
    
    # Add smooth lighting gradient (vignette / ambient light)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    vignette = 1.0 - 0.15 * (xx**2 + yy**2)
    img = img * vignette
    
    # Add subtle machine part features: chamfered circular center or stamped border
    cv2.circle(img, (size // 2, size // 2), size // 3, (base_val - 15), 2)
    cv2.rectangle(img, (8, 8), (size - 8, size - 8), (base_val + 10), 1)
    
    # Clip and convert to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def inject_defect(img_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """Inject a realistic manufacturing defect into a normal part image."""
    img = img_bgr.copy()
    h, w, _ = img.shape
    defect_type = random.choice(["scratch", "crack", "void_hole", "stain"])
    
    if defect_type == "scratch":
        # Sharp high-contrast line
        pt1 = (random.randint(15, w - 15), random.randint(15, h - 15))
        length = random.randint(20, 50)
        angle = random.uniform(0, 2 * np.pi)
        pt2 = (
            int(np.clip(pt1[0] + length * np.cos(angle), 5, w - 5)),
            int(np.clip(pt1[1] + length * np.sin(angle), 5, h - 5)),
        )
        color = random.choice([(30, 30, 30), (240, 240, 240)])
        thickness = random.randint(1, 2)
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        
    elif defect_type == "crack":
        # Branching jagged crack
        curr_x, curr_y = random.randint(20, w - 20), random.randint(20, h - 20)
        num_segments = random.randint(4, 7)
        for _ in range(num_segments):
            next_x = int(np.clip(curr_x + random.randint(-12, 12), 5, w - 5))
            next_y = int(np.clip(curr_y + random.randint(-12, 12), 5, h - 5))
            cv2.line(img, (curr_x, curr_y), (next_x, next_y), (25, 25, 25), 2, cv2.LINE_AA)
            curr_x, curr_y = next_x, next_y
            
    elif defect_type == "void_hole":
        # Dark pit or missing material void
        center = (random.randint(25, w - 25), random.randint(25, h - 25))
        radius = random.randint(4, 9)
        cv2.circle(img, center, radius, (20, 20, 20), -1)
        cv2.circle(img, center, radius + 2, (100, 100, 100), 1)
        
    elif defect_type == "stain":
        # Discoloration / chemical stain patch
        center = (random.randint(25, w - 25), random.randint(25, h - 25))
        axes = (random.randint(8, 16), random.randint(5, 12))
        angle = random.randint(0, 180)
        overlay = img.copy()
        cv2.ellipse(overlay, center, axes, angle, 0, 360, (60, 80, 110), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
    return img, defect_type


def create_inspection_dataset(
    output_dir: str = "data/raw",
    num_train_normal: int = 120,
    num_test_normal: int = 30,
    num_test_defective: int = 30,
    img_size: int = 128,
    seed: int = 42
):
    """Create and save the entire inspection dataset split."""
    random.seed(seed)
    np.random.seed(seed)
    
    base_path = Path(output_dir)
    train_normal_dir = base_path / "train" / "normal"
    test_normal_dir = base_path / "test" / "normal"
    test_defective_dir = base_path / "test" / "defective"
    
    for d in [train_normal_dir, test_normal_dir, test_defective_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    print(f"Generating {num_train_normal} normal training samples...")
    for i in range(num_train_normal):
        img = generate_base_metal_surface(size=img_size)
        cv2.imwrite(str(train_normal_dir / f"normal_{i:04d}.png"), img)
        
    print(f"Generating {num_test_normal} normal test samples...")
    for i in range(num_test_normal):
        img = generate_base_metal_surface(size=img_size)
        cv2.imwrite(str(test_normal_dir / f"normal_test_{i:04d}.png"), img)
        
    print(f"Generating {num_test_defective} defective test samples...")
    defect_log = []
    for i in range(num_test_defective):
        base_img = generate_base_metal_surface(size=img_size)
        defective_img, defect_type = inject_defect(base_img)
        fname = f"defective_{i:04d}_{defect_type}.png"
        cv2.imwrite(str(test_defective_dir / fname), defective_img)
        defect_log.append({"filename": fname, "defect_type": defect_type})
        
    meta = {
        "dataset_name": "Manufacturing_Edge_Visual_Quality_Control",
        "image_dimensions": [img_size, img_size, 3],
        "train_normal_count": num_train_normal,
        "test_normal_count": num_test_normal,
        "test_defective_count": num_test_defective,
        "defect_types": ["scratch", "crack", "void_hole", "stain"],
    }
    
    with open(base_path / "dataset_info.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"✓ Dataset created successfully in {base_path}")
    print(f"  - Train (Normal only): {num_train_normal}")
    print(f"  - Test Normal: {num_test_normal}")
    print(f"  - Test Defective: {num_test_defective}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Visual Quality Dataset")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--train-normal", type=int, default=120)
    parser.add_argument("--test-normal", type=int, default=30)
    parser.add_argument("--test-defective", type=int, default=30)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    
    create_inspection_dataset(
        output_dir=args.output,
        num_train_normal=args.train_normal,
        num_test_normal=args.test_normal,
        num_test_defective=args.test_defective,
        img_size=args.size
    )
