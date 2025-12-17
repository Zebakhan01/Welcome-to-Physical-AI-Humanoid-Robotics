from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import base64
import cv2
from io import BytesIO
from PIL import Image
from backend.utils.sensors_utils import ImageProcessor, CameraIntrinsics
from backend.utils.logger import logger

router = APIRouter()

class ImageProcessingRequest(BaseModel):
    image_data: str  # Base64 encoded image
    operations: List[str]  # List of operations to perform
    camera_intrinsics: Optional[CameraIntrinsics] = None
    params: Optional[Dict[str, Any]] = {}

class ImageProcessingResponse(BaseModel):
    processed_data: Dict[str, Any]
    success: bool
    message: str

class FeatureDetectionRequest(BaseModel):
    image_data: str  # Base64 encoded image
    method: str = "orb"  # "orb", "shi_tomasi", "harris"
    max_features: int = 100

class FeatureDetectionResponse(BaseModel):
    features: List[Dict[str, float]]  # List of feature points {x, y}
    descriptors: List[float]  # Feature descriptors
    num_features: int
    success: bool

class ObjectDetectionRequest(BaseModel):
    image_data: str  # Base64 encoded image
    method: str = "template_matching"  # "template_matching", "edge_based", "color_based"
    template_data: Optional[str] = None  # For template matching

class ObjectDetectionResponse(BaseModel):
    objects: List[Dict[str, Any]]  # List of detected objects with properties
    num_objects: int
    success: bool

class CameraCalibrationRequest(BaseModel):
    images_data: List[str]  # Multiple images for calibration
    pattern_size: List[int]  # [width, height] of calibration pattern
    square_size: float = 1.0  # Size of calibration pattern squares in cm

class CameraCalibrationResponse(BaseModel):
    intrinsics: Optional[CameraIntrinsics]
    reprojection_error: Optional[float]
    success: bool
    message: str

def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)

        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image")

        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode numpy image array to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")

@router.post("/process-image", response_model=ImageProcessingResponse)
async def process_image(request: ImageProcessingRequest):
    """
    Process an image with specified operations
    """
    try:
        # Decode the image
        image = decode_base64_image(request.image_data)

        processed_data = {}
        operations_performed = []

        for operation in request.operations:
            if operation == "edge_detection":
                edges = ImageProcessor.detect_edges(image)
                processed_data["edges"] = "Edge detection performed"
                operations_performed.append("edge_detection")

            elif operation == "corner_detection":
                corners = ImageProcessor.detect_corners(image)
                processed_data["corners"] = [{"x": int(x), "y": int(y)} for x, y in corners]
                operations_performed.append("corner_detection")

            elif operation == "feature_detection":
                points, descriptors = ImageProcessor.compute_features(image)
                processed_data["features"] = [{"x": int(x), "y": int(y)} for x, y in points]
                processed_data["descriptors"] = descriptors
                operations_performed.append("feature_detection")

            elif operation == "undistort" and request.camera_intrinsics:
                undistorted = ImageProcessor.undistort_image(image, request.camera_intrinsics)
                processed_data["undistorted"] = "Image undistorted using intrinsics"
                operations_performed.append("undistort")

            else:
                logger.warning(f"Unknown operation requested: {operation}")

        response = ImageProcessingResponse(
            processed_data=processed_data,
            success=True,
            message=f"Successfully performed {len(operations_performed)} operations: {', '.join(operations_performed)}"
        )

        logger.info(f"Processed image with operations: {operations_performed}")

        return response

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/detect-features", response_model=FeatureDetectionResponse)
async def detect_features(request: FeatureDetectionRequest):
    """
    Detect features in an image
    """
    try:
        # Decode the image
        image = decode_base64_image(request.image_data)

        # Detect features based on method
        points, descriptors = ImageProcessor.compute_features(image, request.method)

        # Limit to max_features if specified
        if len(points) > request.max_features:
            points = points[:request.max_features]
            descriptors = descriptors[:request.max_features]

        features = [{"x": float(x), "y": float(y)} for x, y in points]

        response = FeatureDetectionResponse(
            features=features,
            descriptors=descriptors,
            num_features=len(features),
            success=True
        )

        logger.info(f"Detected {len(features)} features using {request.method} method")

        return response

    except Exception as e:
        logger.error(f"Error detecting features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting features: {str(e)}")

@router.post("/detect-objects", response_model=ObjectDetectionResponse)
async def detect_objects(request: ObjectDetectionRequest):
    """
    Detect objects in an image using various methods
    """
    try:
        # Decode the image
        image = decode_base64_image(request.image_data)

        objects = []

        if request.method == "template_matching" and request.template_data:
            # Decode template
            template = decode_base64_image(request.template_data)

            # Perform template matching
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

            result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.7)  # Threshold for matches

            for pt in zip(*locations[::-1]):
                objects.append({
                    "x": int(pt[0]),
                    "y": int(pt[1]),
                    "width": template.shape[1],
                    "height": template.shape[0],
                    "confidence": float(result[pt[1], pt[0]])
                })

        elif request.method == "edge_based":
            # Simple edge-based detection (looking for rectangular objects)
            edges = ImageProcessor.detect_edges(image)

            # Find contours
            contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # If it's approximately rectangular
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h

                    if 0.5 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio
                        objects.append({
                            "x": int(x),
                            "y": int(y),
                            "width": int(w),
                            "height": int(h),
                            "type": "rectangular_object",
                            "confidence": 0.8
                        })

        elif request.method == "color_based":
            # Simple color-based detection (looking for red objects as example)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define range for red color
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2

            # Find contours of red regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "type": "red_object",
                        "confidence": 0.7
                    })

        response = ObjectDetectionResponse(
            objects=objects,
            num_objects=len(objects),
            success=True
        )

        logger.info(f"Detected {len(objects)} objects using {request.method} method")

        return response

    except Exception as e:
        logger.error(f"Error detecting objects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting objects: {str(e)}")

@router.post("/calibrate-camera", response_model=CameraCalibrationResponse)
async def calibrate_camera(request: CameraCalibrationRequest):
    """
    Calibrate camera using multiple images of a calibration pattern
    """
    try:
        if len(request.images_data) < 3:
            raise HTTPException(status_code=400, detail="At least 3 images required for calibration")

        if len(request.pattern_size) != 2:
            raise HTTPException(status_code=400, detail="Pattern size must be [width, height]")

        # Prepare object points (3D points of calibration pattern)
        objp = np.zeros((request.pattern_size[0] * request.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:request.pattern_size[0], 0:request.pattern_size[1]].T.reshape(-1, 2)
        objp *= request.square_size  # Scale by square size

        # Arrays to store object points and image points from all images
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane

        # Process each calibration image
        for img_data in request.images_data:
            image = decode_base64_image(img_data)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, tuple(request.pattern_size), None)

            # If found, add object points and image points
            if ret:
                obj_points.append(objp)
                # Refine corner locations
                refined_corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                img_points.append(refined_corners)

        if len(obj_points) < 3:
            return CameraCalibrationResponse(
                intrinsics=None,
                reprojection_error=None,
                success=False,
                message="Could not find calibration pattern in enough images"
            )

        # Perform camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )

        if not ret:
            return CameraCalibrationResponse(
                intrinsics=None,
                reprojection_error=None,
                success=False,
                message="Camera calibration failed"
            )

        # Create CameraIntrinsics object
        intrinsics = CameraIntrinsics(
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]),
            cy=float(camera_matrix[1, 2]),
            k1=float(dist_coeffs[0, 0]) if dist_coeffs.shape[0] > 0 else 0.0,
            k2=float(dist_coeffs[0, 1]) if dist_coeffs.shape[0] > 1 else 0.0,
            p1=float(dist_coeffs[0, 2]) if dist_coeffs.shape[0] > 2 else 0.0,
            p2=float(dist_coeffs[0, 3]) if dist_coeffs.shape[0] > 3 else 0.0
        )

        response = CameraCalibrationResponse(
            intrinsics=intrinsics,
            reprojection_error=float(ret) if ret else None,
            success=True,
            message=f"Successfully calibrated camera using {len(obj_points)} images"
        )

        logger.info(f"Camera calibrated using {len(obj_points)} images")

        return response

    except Exception as e:
        logger.error(f"Error calibrating camera: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calibrating camera: {str(e)}")

class StereoVisionRequest(BaseModel):
    left_image: str  # Base64 encoded left camera image
    right_image: str  # Base64 encoded right camera image
    baseline: float  # Distance between cameras in meters
    focal_length: float  # Focal length in pixels

class StereoVisionResponse(BaseModel):
    disparity_map: Optional[str]  # Base64 encoded disparity map
    depth_map: Optional[str]  # Base64 encoded depth map
    objects_3d: List[Dict[str, float]]  # 3D object coordinates
    success: bool

@router.post("/stereo-vision", response_model=StereoVisionResponse)
async def stereo_vision(request: StereoVisionRequest):
    """
    Perform stereo vision processing to extract depth information
    """
    try:
        # Decode images
        left_image = decode_base64_image(request.left_image)
        right_image = decode_base64_image(request.right_image)

        # Convert to grayscale
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY) if len(left_image.shape) == 3 else left_image
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY) if len(right_image.shape) == 3 else right_image

        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,  # Must be divisible by 16
            blockSize=15,
            P1=8 * 3 * 15**2,
            P2=32 * 3 * 15**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute disparity map
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Convert disparity to depth (simplified formula: depth = (baseline * focal) / disparity)
        depth_map = np.zeros_like(disparity)
        valid_disparity = disparity > 0
        depth_map[valid_disparity] = (request.baseline * request.focal_length) / disparity[valid_disparity]

        # Find objects in the depth map (simplified approach)
        objects_3d = []

        # Simple approach: find regions with similar depth
        for y in range(0, depth_map.shape[0], 20):  # Sample every 20 pixels
            for x in range(0, depth_map.shape[1], 20):
                depth_val = depth_map[y, x]
                if depth_val > 0 and depth_val < 100:  # Valid depth range
                    # Convert pixel coordinates to 3D world coordinates (simplified)
                    z = float(depth_val)
                    x_3d = (x - left_gray.shape[1]/2) * z / request.focal_length
                    y_3d = (y - left_gray.shape[0]/2) * z / request.focal_length

                    objects_3d.append({
                        "x": x_3d,
                        "y": y_3d,
                        "z": z,
                        "pixel_x": x,
                        "pixel_y": y
                    })

        # For now, return success without encoded maps to avoid large data transfer
        response = StereoVisionResponse(
            disparity_map=None,  # Would be encoded disparity map
            depth_map=None,      # Would be encoded depth map
            objects_3d=objects_3d[:50],  # Limit to first 50 objects
            success=True
        )

        logger.info(f"Stereo vision processing completed, found {len(objects_3d)} 3D points")

        return response

    except Exception as e:
        logger.error(f"Error in stereo vision processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in stereo vision processing: {str(e)}")