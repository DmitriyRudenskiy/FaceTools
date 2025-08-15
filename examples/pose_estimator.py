"""
Human Face Pose Estimator
=========================

This script provides human face pose estimation capabilities using DWPose.
It detects and estimates poses only for human faces.

Installation:
    pip install opencv-python torch torchvision onnxruntime
    pip install git+https://github.com/IDEA-Research/DWPose.git

Usage:
    python /Users/user/PycharmProjects/FaceCluster/examples/console_dw_open_pose.py --input /Users/user/Downloads/f31a8051d446bd94331f8d4b33b39951.jpg --output face_pose_output.jpg
"""

import argparse
import cv2
import torch
import numpy as np
import json
import os
import warnings
from typing import Tuple, Dict, Any, Optional

# Try to import DWPose components
try:
    from custom_controlnet_aux.dwpose import DwposeDetector

    DWPose_AVAILABLE = True
except ImportError:
    DWPose_AVAILABLE = False
    print("Warning: DWPose not available. Please install required dependencies.")

DWPOSE_MODEL_NAME = "yzd-v/DWPose"


def check_ort_gpu():
    """Check if ONNX Runtime has GPU acceleration providers available."""
    try:
        import onnxruntime as ort
        GPU_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider",
                         "OpenVINOExecutionProvider", "ROCMExecutionProvider",
                         "CoreMLExecutionProvider"]
        for provider in GPU_PROVIDERS:
            if provider in ort.get_available_providers():
                return True
        return False
    except:
        return False


def setup_environment():
    """Setup environment variables and check dependencies."""
    if not os.environ.get("DWPOSE_ONNXRT_CHECKED"):
        if check_ort_gpu():
            print("DWPose: Onnxruntime with acceleration providers detected")
        else:
            warnings.warn(
                "DWPose: Onnxruntime not found or doesn't come with acceleration providers, switch to OpenCV with CPU device. DWPose might run very slowly")
            os.environ['AUX_ORT_PROVIDERS'] = ''
        os.environ["DWPOSE_ONNXRT_CHECKED"] = '1'


class FacePoseEstimator:
    """Main face pose estimation class supporting only human face detection."""

    def __init__(self):
        setup_environment()
        self.openpose_dicts = []

    def estimate_face_pose(self,
                          image_path: str,
                          resolution: int = 512,
                          bbox_detector: str = "yolox_l.onnx",
                          pose_estimator: str = "dw-ll_ucoco_384_bs5.torchscript.pt") -> Tuple[np.ndarray, Dict]:
        """
        Estimate human face pose using DWPose.

        Args:
            image_path: Path to input image
            resolution: Output resolution
            bbox_detector: Bounding box detector model
            pose_estimator: Pose estimation model

        Returns:
            Tuple of (pose_image, keypoints_dict)
        """
        if not DWPose_AVAILABLE:
            raise RuntimeError("DWPose is not available. Please install required dependencies.")

        # Determine repositories based on model names
        if bbox_detector == "yolox_l.onnx":
            yolo_repo = DWPOSE_MODEL_NAME
        elif "yolox" in bbox_detector:
            yolo_repo = "hr16/yolox-onnx"
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
        else:
            raise NotImplementedError(f"Download mechanism for {bbox_detector}")

        if pose_estimator == "dw-ll_ucoco_384.onnx":
            pose_repo = DWPOSE_MODEL_NAME
        elif pose_estimator.endswith(".onnx"):
            pose_repo = "hr16/UnJIT-DWPose"
        elif pose_estimator.endswith(".torchscript.pt"):
            pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
        else:
            raise NotImplementedError(f"Download mechanism for {pose_estimator}")

        # Load model
        model = DwposeDetector.from_pretrained(
            pose_repo,
            yolo_repo,
            det_filename=bbox_detector,
            pose_filename=pose_estimator,
            torchscript_device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Resize image to target resolution
        h, w = image.shape[:2]
        scale = resolution / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = cv2.resize(image, (new_w, new_h))

        # Estimate pose - only face detection enabled
        pose_img, openpose_dict = model(
            image_resized,
            include_hand=False,
            include_face=True,
            include_body=False
        )

        self.openpose_dicts.append(openpose_dict)
        del model

        return pose_img, openpose_dict

    def save_results(self, pose_image: np.ndarray, keypoints_dict: Dict, output_path: str,
                     json_path: Optional[str] = None):
        """
        Save face pose estimation results.

        Args:
            pose_image: Generated pose image
            keypoints_dict: Keypoints dictionary
            output_path: Path to save pose image
            json_path: Path to save keypoints JSON (optional)
        """
        # Save pose image
        cv2.imwrite(output_path, pose_image)
        print(f"Face pose image saved to {output_path}")

        # Save keypoints JSON
        if json_path is None:
            json_path = output_path.replace('.jpg', '.json').replace('.png', '.json')

        with open(json_path, 'w') as f:
            json.dump(keypoints_dict, f, indent=4)
        print(f"Face keypoints JSON saved to {json_path}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Human Face Pose Estimation using DWPose")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    parser.add_argument("--resolution", "-r", type=int, default=512, help="Output resolution")
    parser.add_argument("--bbox-detector", default="yolox_l.onnx", help="Bounding box detector model")
    parser.add_argument("--pose-estimator", default="dw-ll_ucoco_384_bs5.torchscript.pt", help="Pose estimator model")
    parser.add_argument("--json-output", help="Output JSON path for keypoints")

    args = parser.parse_args()

    # Create estimator
    estimator = FacePoseEstimator()

    try:
        # Estimate face pose only
        pose_img, keypoints = estimator.estimate_face_pose(
            image_path=args.input,
            resolution=args.resolution,
            bbox_detector=args.bbox_detector,
            pose_estimator=args.pose_estimator
        )

        # Save results
        estimator.save_results(pose_img, keypoints, args.output, args.json_output)
        print("Face pose estimation completed successfully!")

    except Exception as e:
        print(f"Error during face pose estimation: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())