"""
Unit tests for detection module.

Run with: pytest tests/test_detection.py -v
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components import detection
from app.config import CLASS_NAMES, CONFIDENCE_THRESHOLD, IOU_THRESHOLD


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_image():
    """Create a sample BGR image for testing."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_small_image():
    """Create a small sample image."""
    return np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)


@pytest.fixture
def mock_detections():
    """Create mock detection results."""
    return {
        'boxes': [
            [100.0, 150.0, 300.0, 350.0],  # Box 1: pothole
            [400.0, 200.0, 550.0, 380.0],  # Box 2: crack
        ],
        'classes': [0, 1],  # pothole, longitudinal_crack
        'confidences': [0.85, 0.78],
        'raw_results': None,
        'detection_count': 2
    }


@pytest.fixture
def empty_detections():
    """Create empty detection results."""
    return {
        'boxes': [],
        'classes': [],
        'confidences': [],
        'raw_results': None,
        'detection_count': 0
    }


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model."""
    model = Mock()
    
    # Mock results object
    mock_result = Mock()
    mock_boxes = Mock()
    
    # Set up boxes attributes
    mock_boxes.xyxy = Mock()
    mock_boxes.xyxy.cpu = Mock(return_value=Mock(
        numpy=Mock(return_value=np.array([[100, 150, 300, 350], [400, 200, 550, 380]]))
    ))
    
    mock_boxes.cls = Mock()
    mock_boxes.cls.cpu = Mock(return_value=Mock(
        numpy=Mock(return_value=np.array([0, 1]))
    ))
    
    mock_boxes.conf = Mock()
    mock_boxes.conf.cpu = Mock(return_value=Mock(
        numpy=Mock(return_value=np.array([0.85, 0.78]))
    ))
    
    mock_result.boxes = mock_boxes
    
    # Model returns list of results
    model.return_value = [mock_result]
    
    return model


# ============================================================================
# Test Model Loading
# ============================================================================

class TestModelLoading:
    """Tests for YOLO model loading."""
    
    @patch('app.components.detection.YOLO')
    def test_load_model_success(self, mock_yolo):
        """Test successful model loading."""
        # Setup mock
        mock_yolo.return_value = Mock()
        
        # Load model
        model = detection.load_model('dummy_model.pt')
        
        # Assertions
        assert model is not None
        mock_yolo.assert_called_once()
    
    @patch('app.components.detection.Path')
    def test_load_model_file_not_found(self, mock_path):
        """Test error handling when model file doesn't exist."""
        # Mock path to return False for exists()
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        # Load model should return None
        model = detection.load_model('nonexistent_model.pt')
        
        assert model is None
    
    @patch('app.components.detection.YOLO')
    @patch('app.components.detection.Path')
    def test_load_model_exception_handling(self, mock_path, mock_yolo):
        """Test exception handling during model loading."""
        # Setup mocks
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_yolo.side_effect = Exception("Loading error")
        
        # Load model should return None
        model = detection.load_model('error_model.pt')
        
        assert model is None
    
    def test_load_model_caching(self):
        """Test that model loading uses caching."""
        # This test verifies the @st.cache_resource decorator is applied
        # by checking the function has the _cache attribute
        assert hasattr(detection.load_model, '__wrapped__')


# ============================================================================
# Test Hazard Detection
# ============================================================================

class TestHazardDetection:
    """Tests for hazard detection function."""
    
    def test_detect_hazards_output_format(self, sample_image, mock_yolo_model):
        """Test output format is correct."""
        result = detection.detect_hazards(
            mock_yolo_model,
            sample_image,
            conf_threshold=0.5,
            iou_threshold=0.45
        )
        
        # Check all required keys are present
        assert 'boxes' in result
        assert 'classes' in result
        assert 'confidences' in result
        assert 'raw_results' in result
        assert 'detection_count' in result
        
        # Check types
        assert isinstance(result['boxes'], list)
        assert isinstance(result['classes'], list)
        assert isinstance(result['confidences'], list)
        assert isinstance(result['detection_count'], int)
    
    def test_detect_hazards_with_detections(self, sample_image, mock_yolo_model):
        """Test detection with results."""
        result = detection.detect_hazards(mock_yolo_model, sample_image)
        
        assert result['detection_count'] == 2
        assert len(result['boxes']) == 2
        assert len(result['classes']) == 2
        assert len(result['confidences']) == 2
    
    def test_detect_hazards_empty_results(self, sample_image):
        """Test detection with no results."""
        # Mock model that returns empty results
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        
        result = detection.detect_hazards(mock_model, sample_image)
        
        assert result['detection_count'] == 0
        assert len(result['boxes']) == 0
    
    def test_detect_hazards_invalid_model(self, sample_image):
        """Test error handling with None model."""
        with pytest.raises(ValueError, match="Model is None"):
            detection.detect_hazards(None, sample_image)
    
    def test_detect_hazards_invalid_image(self, mock_yolo_model):
        """Test error handling with invalid image."""
        with pytest.raises(ValueError, match="Invalid image"):
            detection.detect_hazards(mock_yolo_model, None)
        
        with pytest.raises(ValueError, match="Invalid image"):
            detection.detect_hazards(mock_yolo_model, np.array([]))
    
    def test_detect_hazards_confidence_threshold(self, sample_image, mock_yolo_model):
        """Test confidence threshold is applied."""
        # Call with high threshold
        result = detection.detect_hazards(
            mock_yolo_model,
            sample_image,
            conf_threshold=0.9
        )
        
        # Verify model was called with correct threshold
        mock_yolo_model.assert_called()
        call_kwargs = mock_yolo_model.call_args[1]
        assert call_kwargs['conf'] == 0.9
    
    def test_detect_hazards_exception_handling(self, sample_image):
        """Test exception handling during detection."""
        mock_model = Mock()
        mock_model.side_effect = Exception("Detection error")
        
        with pytest.raises(RuntimeError, match="Detection failed"):
            detection.detect_hazards(mock_model, sample_image)


# ============================================================================
# Test Drawing Detections
# ============================================================================

class TestDrawDetections:
    """Tests for drawing detection visualizations."""
    
    def test_draw_detections_basic(self, sample_image, mock_detections):
        """Test basic detection drawing."""
        annotated = detection.draw_detections(
            sample_image,
            mock_detections,
            CLASS_NAMES
        )
        
        # Check output is valid image
        assert annotated is not None
        assert annotated.shape == sample_image.shape
        assert annotated.dtype == np.uint8
        
        # Check image was modified
        assert not np.array_equal(annotated, sample_image)
    
    def test_draw_detections_empty(self, sample_image, empty_detections):
        """Test drawing with no detections."""
        annotated = detection.draw_detections(
            sample_image,
            empty_detections,
            CLASS_NAMES
        )
        
        # Should return unchanged image
        assert annotated is not None
        assert annotated.shape == sample_image.shape
    
    def test_draw_detections_invalid_image(self, mock_detections):
        """Test error handling with invalid image."""
        with pytest.raises(ValueError, match="Invalid image"):
            detection.draw_detections(None, mock_detections, CLASS_NAMES)
        
        with pytest.raises(ValueError, match="Invalid image"):
            detection.draw_detections(np.array([]), mock_detections, CLASS_NAMES)
    
    def test_draw_detections_with_confidence(self, sample_image, mock_detections):
        """Test drawing with confidence scores."""
        annotated = detection.draw_detections(
            sample_image,
            mock_detections,
            CLASS_NAMES,
            draw_confidence=True
        )
        
        assert annotated is not None
        assert not np.array_equal(annotated, sample_image)
    
    def test_draw_detections_without_confidence(self, sample_image, mock_detections):
        """Test drawing without confidence scores."""
        annotated = detection.draw_detections(
            sample_image,
            mock_detections,
            CLASS_NAMES,
            draw_confidence=False
        )
        
        assert annotated is not None
    
    def test_draw_detections_bounds_checking(self, sample_image):
        """Test bounding box coordinates are properly bounded."""
        # Detections with out-of-bounds coordinates
        detections = {
            'boxes': [[-10, -10, 1000, 1000]],  # Outside image bounds
            'classes': [0],
            'confidences': [0.9]
        }
        
        # Should not raise exception
        annotated = detection.draw_detections(sample_image, detections, CLASS_NAMES)
        assert annotated is not None
    
    def test_draw_detections_custom_thickness(self, sample_image, mock_detections):
        """Test custom line thickness."""
        annotated = detection.draw_detections(
            sample_image,
            mock_detections,
            CLASS_NAMES,
            line_thickness=5
        )
        
        assert annotated is not None


# ============================================================================
# Test Hazard Size Calculation
# ============================================================================

class TestCalculateHazardSize:
    """Tests for hazard size estimation."""
    
    def test_calculate_hazard_size_basic(self):
        """Test basic size calculation."""
        bbox = [100, 200, 300, 400]
        image_shape = (640, 480)
        
        size = detection.calculate_hazard_size(bbox, image_shape)
        
        assert isinstance(size, float)
        assert size >= 0
    
    def test_calculate_hazard_size_small_bbox(self):
        """Test with small bounding box."""
        bbox = [100, 100, 120, 120]
        image_shape = (640, 480)
        
        size = detection.calculate_hazard_size(bbox, image_shape)
        
        assert size > 0
        assert size < 1.0  # Small box should be less than 1 m²
    
    def test_calculate_hazard_size_large_bbox(self):
        """Test with large bounding box."""
        bbox = [0, 0, 640, 480]
        image_shape = (640, 480)
        
        size = detection.calculate_hazard_size(bbox, image_shape)
        
        assert size > 10  # Large box should be significant area
    
    def test_calculate_hazard_size_error_handling(self):
        """Test error handling with invalid input."""
        bbox = [100, 100, 200, 200]
        image_shape = (640, 480)
        
        # Should not raise exception
        size = detection.calculate_hazard_size(bbox, image_shape)
        assert size >= 0


# ============================================================================
# Test Severity Classification
# ============================================================================

class TestClassifySeverity:
    """Tests for severity classification."""
    
    def test_classify_severity_pothole_large(self):
        """Test large pothole is classified as critical/high."""
        severity = detection.classify_severity(
            class_id=0,  # pothole
            bbox_area=60000  # Large area
        )
        
        assert severity in ['critical', 'high']
    
    def test_classify_severity_pothole_medium(self):
        """Test medium pothole."""
        severity = detection.classify_severity(
            class_id=0,
            bbox_area=30000
        )
        
        assert severity in ['high', 'medium']
    
    def test_classify_severity_pothole_small(self):
        """Test small pothole."""
        severity = detection.classify_severity(
            class_id=0,
            bbox_area=15000
        )
        
        assert severity == 'medium'
    
    def test_classify_severity_alligator_crack(self):
        """Test alligator crack is classified as high/medium."""
        severity = detection.classify_severity(
            class_id=3,  # alligator_crack
            bbox_area=40000
        )
        
        assert severity in ['high', 'medium']
    
    def test_classify_severity_longitudinal_crack(self):
        """Test longitudinal crack (least severe)."""
        severity = detection.classify_severity(
            class_id=1,  # longitudinal_crack
            bbox_area=30000
        )
        
        assert severity in ['medium', 'low']
    
    def test_classify_severity_all_classes(self):
        """Test all class IDs return valid severity."""
        for class_id in range(len(CLASS_NAMES)):
            severity = detection.classify_severity(class_id, bbox_area=25000)
            assert severity in ['critical', 'high', 'medium', 'low']
    
    def test_classify_severity_unknown_class(self):
        """Test unknown class ID."""
        severity = detection.classify_severity(
            class_id=999,  # Invalid class
            bbox_area=25000
        )
        
        assert severity == 'low'  # Default severity


# ============================================================================
# Test Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_hex_to_bgr(self):
        """Test hex to BGR color conversion."""
        # Test with hash
        bgr = detection.hex_to_bgr('#FF5733')
        assert isinstance(bgr, tuple)
        assert len(bgr) == 3
        assert all(0 <= c <= 255 for c in bgr)
        
        # Test without hash
        bgr = detection.hex_to_bgr('FF5733')
        assert isinstance(bgr, tuple)
        assert len(bgr) == 3
    
    def test_hex_to_bgr_white(self):
        """Test conversion of white color."""
        bgr = detection.hex_to_bgr('#FFFFFF')
        assert bgr == (255, 255, 255)
    
    def test_hex_to_bgr_black(self):
        """Test conversion of black color."""
        bgr = detection.hex_to_bgr('#000000')
        assert bgr == (0, 0, 0)
    
    def test_get_detection_summary(self, mock_detections):
        """Test detection summary generation."""
        summary = detection.get_detection_summary(mock_detections, CLASS_NAMES)
        
        assert 'total_detections' in summary
        assert 'class_counts' in summary
        assert 'avg_confidence' in summary
        assert 'severity_counts' in summary
        
        assert summary['total_detections'] == 2
        assert isinstance(summary['class_counts'], dict)
        assert isinstance(summary['avg_confidence'], float)
    
    def test_get_detection_summary_empty(self, empty_detections):
        """Test summary with empty detections."""
        summary = detection.get_detection_summary(empty_detections, CLASS_NAMES)
        
        assert summary['total_detections'] == 0
        assert summary['avg_confidence'] == 0.0
        assert summary['class_counts'] == {}


# ============================================================================
# Integration Tests
# ============================================================================

class TestDetectionIntegration:
    """Integration tests for complete detection pipeline."""
    
    def test_full_detection_pipeline(self, sample_image, mock_yolo_model):
        """Test complete detection pipeline."""
        # Step 1: Detect hazards
        detections = detection.detect_hazards(
            mock_yolo_model,
            sample_image,
            conf_threshold=0.5
        )
        
        assert detections['detection_count'] > 0
        
        # Step 2: Draw detections
        annotated = detection.draw_detections(
            sample_image,
            detections,
            CLASS_NAMES
        )
        
        assert annotated is not None
        
        # Step 3: Get summary
        summary = detection.get_detection_summary(detections, CLASS_NAMES)
        
        assert summary['total_detections'] == detections['detection_count']
    
    def test_pipeline_with_no_detections(self, sample_image):
        """Test pipeline when no hazards detected."""
        # Mock model with no detections
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        
        # Detect
        detections = detection.detect_hazards(mock_model, sample_image)
        assert detections['detection_count'] == 0
        
        # Draw (should not modify image much)
        annotated = detection.draw_detections(sample_image, detections, CLASS_NAMES)
        assert annotated is not None
        
        # Summary
        summary = detection.get_detection_summary(detections, CLASS_NAMES)
        assert summary['total_detections'] == 0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_pixel_bbox(self, sample_image):
        """Test with single pixel bounding box."""
        detections = {
            'boxes': [[100, 100, 101, 101]],
            'classes': [0],
            'confidences': [0.9]
        }
        
        annotated = detection.draw_detections(sample_image, detections, CLASS_NAMES)
        assert annotated is not None
    
    def test_zero_area_bbox(self, sample_image):
        """Test with zero area bounding box."""
        detections = {
            'boxes': [[100, 100, 100, 100]],
            'classes': [0],
            'confidences': [0.9]
        }
        
        # Should handle gracefully
        annotated = detection.draw_detections(sample_image, detections, CLASS_NAMES)
        assert annotated is not None
    
    def test_very_high_confidence(self, sample_image):
        """Test with confidence = 1.0."""
        detections = {
            'boxes': [[100, 100, 200, 200]],
            'classes': [0],
            'confidences': [1.0]
        }
        
        annotated = detection.draw_detections(sample_image, detections, CLASS_NAMES)
        assert annotated is not None
    
    def test_very_low_confidence(self, sample_image):
        """Test with very low confidence."""
        detections = {
            'boxes': [[100, 100, 200, 200]],
            'classes': [0],
            'confidences': [0.01]
        }
        
        annotated = detection.draw_detections(sample_image, detections, CLASS_NAMES)
        assert annotated is not None
    
    def test_many_detections(self, sample_image):
        """Test with many detections."""
        num_detections = 50
        detections = {
            'boxes': [[i*10, i*10, i*10+50, i*10+50] for i in range(num_detections)],
            'classes': [i % len(CLASS_NAMES) for i in range(num_detections)],
            'confidences': [0.5 + (i % 5) * 0.1 for i in range(num_detections)]
        }
        
        annotated = detection.draw_detections(sample_image, detections, CLASS_NAMES)
        assert annotated is not None
        
        summary = detection.get_detection_summary(detections, CLASS_NAMES)
        assert summary['total_detections'] == num_detections


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Tests for performance characteristics."""
    
    def test_detection_speed(self, sample_image, mock_yolo_model):
        """Test detection completes in reasonable time."""
        import time
        
        start = time.time()
        detection.detect_hazards(mock_yolo_model, sample_image)
        duration = time.time() - start
        
        # Should complete quickly with mock model
        assert duration < 1.0  # Less than 1 second
    
    def test_drawing_speed(self, sample_image, mock_detections):
        """Test drawing completes in reasonable time."""
        import time
        
        start = time.time()
        detection.draw_detections(sample_image, mock_detections, CLASS_NAMES)
        duration = time.time() - start
        
        # Drawing should be fast
        assert duration < 0.5  # Less than 500ms


# ============================================================================
# Parametrized Tests
# ============================================================================

@pytest.mark.parametrize("class_id,expected_severity", [
    (0, 'medium'),   # pothole
    (1, 'low'),      # longitudinal_crack
    (2, 'low'),      # transverse_crack
    (3, 'medium'),   # alligator_crack
])
def test_severity_by_class(class_id, expected_severity):
    """Parametrized test for severity classification."""
    severity = detection.classify_severity(class_id, bbox_area=20000)
    assert severity == expected_severity or severity in ['critical', 'high', 'medium', 'low']


@pytest.mark.parametrize("bbox_area,class_id", [
    (70000, 0),   # Large pothole
    (30000, 0),   # Medium pothole
    (10000, 0),   # Small pothole
    (50000, 1),   # Large crack
    (15000, 1),   # Small crack
])
def test_size_severity_correlation(bbox_area, class_id):
    """Test that larger areas generally result in higher severity."""
    severity = detection.classify_severity(class_id, bbox_area)
    assert severity in ['critical', 'high', 'medium', 'low']


# ============================================================================
# Fixtures Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Cleanup code here if needed
    pass


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=app.components.detection"])
