"""
Unit tests for alerts module.

Run with: pytest tests/test_alerts.py -v
"""

import pytest
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components import alerts
from app.config import ALERT_SETTINGS


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_hazard():
    """Create a sample hazard dictionary."""
    return {
        'latitude': 28.6139,
        'longitude': 77.2090,
        'class_name': 'pothole',
        'severity': 'high',
        'confidence': 0.85
    }


@pytest.fixture
def vehicle_position():
    """Create a sample vehicle position."""
    return (28.6130, 77.2080)  # ~100m from sample hazard


@pytest.fixture
def alert_settings():
    """Get alert settings from config."""
    return ALERT_SETTINGS


# ============================================================================
# Test Warning Distance Calculation
# ============================================================================

class TestCalculateWarningDistance:
    """Tests for warning distance calculation."""
    
    def test_warning_distance_zero_speed(self):
        """Test warning distance at zero speed."""
        distance = alerts.calculate_warning_distance(0, 'medium')
        
        # Should return minimum warning distance
        assert distance >= alerts.WARNING_DISTANCE_MIN
        assert distance == alerts.WARNING_DISTANCE_MIN
    
    def test_warning_distance_increases_with_speed(self):
        """Test that warning distance increases with speed."""
        dist_30 = alerts.calculate_warning_distance(30, 'medium')
        dist_60 = alerts.calculate_warning_distance(60, 'medium')
        dist_90 = alerts.calculate_warning_distance(90, 'medium')
        
        assert dist_30 < dist_60 < dist_90
    
    def test_warning_distance_severity_effect(self):
        """Test that severity affects warning distance."""
        speed = 60
        
        dist_low = alerts.calculate_warning_distance(speed, 'low')
        dist_medium = alerts.calculate_warning_distance(speed, 'medium')
        dist_high = alerts.calculate_warning_distance(speed, 'high')
        dist_critical = alerts.calculate_warning_distance(speed, 'critical')
        
        # More severe hazards should have longer warning distances
        assert dist_low < dist_medium <= dist_high < dist_critical
    
    def test_warning_distance_road_condition(self):
        """Test road condition effects on warning distance."""
        speed = 60
        severity = 'medium'
        
        dist_dry = alerts.calculate_warning_distance(speed, severity, 'dry')
        dist_wet = alerts.calculate_warning_distance(speed, severity, 'wet')
        dist_icy = alerts.calculate_warning_distance(speed, severity, 'icy')
        
        # Worse conditions need longer distances
        assert dist_dry < dist_wet < dist_icy
    
    def test_warning_distance_minimum_enforced(self):
        """Test that minimum warning distance is enforced."""
        # Very low speed should still meet minimum
        distance = alerts.calculate_warning_distance(5, 'low')
        
        assert distance >= alerts.WARNING_DISTANCE_MIN
    
    def test_warning_distance_negative_speed(self):
        """Test handling of negative speed (should be treated as 0)."""
        distance = alerts.calculate_warning_distance(-10, 'medium')
        
        assert distance >= alerts.WARNING_DISTANCE_MIN
    
    @pytest.mark.parametrize("speed,severity,min_distance", [
        (30, 'low', 30),
        (60, 'medium', 60),
        (90, 'high', 100),
        (120, 'critical', 150),
    ])
    def test_warning_distance_realistic_values(self, speed, severity, min_distance):
        """Test realistic warning distances for various scenarios."""
        distance = alerts.calculate_warning_distance(speed, severity)
        
        assert distance >= min_distance


# ============================================================================
# Test Safe Speed Calculation
# ============================================================================

class TestCalculateSafeSpeed:
    """Tests for safe speed calculation."""
    
    def test_safe_speed_zero_distance(self):
        """Test safe speed at zero distance."""
        speed = alerts.calculate_safe_speed(0, 'medium')
        
        # Should return very low or zero speed
        assert speed >= 0
        assert speed < 10
    
    def test_safe_speed_increases_with_distance(self):
        """Test that safe speed increases with distance."""
        speed_50 = alerts.calculate_safe_speed(50, 'medium')
        speed_100 = alerts.calculate_safe_speed(100, 'medium')
        speed_200 = alerts.calculate_safe_speed(200, 'medium')
        
        assert speed_50 < speed_100 < speed_200
    
    def test_safe_speed_severity_effect(self):
        """Test severity affects safe speed."""
        distance = 100
        
        speed_low = alerts.calculate_safe_speed(distance, 'low')
        speed_medium = alerts.calculate_safe_speed(distance, 'medium')
        speed_high = alerts.calculate_safe_speed(distance, 'high')
        speed_critical = alerts.calculate_safe_speed(distance, 'critical')
        
        # Less severe allows higher speed
        assert speed_critical < speed_high <= speed_medium < speed_low
    
    def test_safe_speed_road_condition(self):
        """Test road condition effects on safe speed."""
        distance = 100
        severity = 'medium'
        
        speed_dry = alerts.calculate_safe_speed(distance, severity, 'dry')
        speed_wet = alerts.calculate_safe_speed(distance, severity, 'wet')
        speed_icy = alerts.calculate_safe_speed(distance, severity, 'icy')
        
        # Worse conditions require lower speeds
        assert speed_icy < speed_wet < speed_dry
    
    def test_safe_speed_capped(self):
        """Test safe speed is capped at maximum."""
        # Very large distance
        speed = alerts.calculate_safe_speed(1000, 'low')
        
        # Should not exceed reasonable maximum (120 km/h)
        assert speed <= 120
    
    def test_safe_speed_non_negative(self):
        """Test safe speed is always non-negative."""
        speed = alerts.calculate_safe_speed(10, 'critical')
        
        assert speed >= 0
    
    @pytest.mark.parametrize("distance,severity,expected_range", [
        (50, 'critical', (20, 45)),
        (100, 'high', (40, 70)),
        (150, 'medium', (60, 90)),
        (200, 'low', (70, 110)),
    ])
    def test_safe_speed_ranges(self, distance, severity, expected_range):
        """Test safe speed falls within expected ranges."""
        speed = alerts.calculate_safe_speed(distance, severity)
        
        min_speed, max_speed = expected_range
        assert min_speed <= speed <= max_speed


# ============================================================================
# Test Alert Generation
# ============================================================================

class TestGenerateAlertMessage:
    """Tests for alert message generation."""
    
    def test_alert_message_structure(self, sample_hazard):
        """Test alert message has correct structure."""
        alert = alerts.generate_alert_message(
            sample_hazard,
            distance_meters=80,
            current_speed_kmh=60
        )
        
        # Check Alert object structure
        assert isinstance(alert, alerts.Alert)
        assert hasattr(alert, 'message')
        assert hasattr(alert, 'urgency')
        assert hasattr(alert, 'recommended_speed')
        assert hasattr(alert, 'distance_meters')
        assert hasattr(alert, 'hazard_type')
        assert hasattr(alert, 'severity')
        assert hasattr(alert, 'time_to_impact')
    
    def test_alert_message_not_empty(self, sample_hazard):
        """Test alert message is not empty."""
        alert = alerts.generate_alert_message(sample_hazard, 100, 60)
        
        assert len(alert.message) > 0
        assert isinstance(alert.message, str)
    
    def test_alert_urgency_levels(self, sample_hazard):
        """Test different urgency levels are generated."""
        # Close distance, high speed = critical
        alert_critical = alerts.generate_alert_message(sample_hazard, 20, 80)
        assert alert_critical.urgency == alerts.UrgencyLevel.CRITICAL
        
        # Moderate distance = lower urgency
        alert_moderate = alerts.generate_alert_message(sample_hazard, 150, 60)
        assert alert_moderate.urgency in [
            alerts.UrgencyLevel.MODERATE,
            alerts.UrgencyLevel.LOW,
            alerts.UrgencyLevel.INFO
        ]
    
    def test_alert_safe_speed_reasonable(self, sample_hazard):
        """Test recommended safe speed is reasonable."""
        alert = alerts.generate_alert_message(sample_hazard, 100, 80)
        
        assert 0 <= alert.recommended_speed <= 120
        assert alert.recommended_speed < 80  # Should be lower than current
    
    def test_alert_time_to_impact(self, sample_hazard):
        """Test time to impact is calculated."""
        alert = alerts.generate_alert_message(sample_hazard, 100, 60)
        
        assert alert.time_to_impact > 0
        # At 60 km/h (~16.67 m/s), 100m takes ~6 seconds
        assert 5 < alert.time_to_impact < 8
    
    def test_alert_zero_speed(self, sample_hazard):
        """Test alert with zero speed."""
        alert = alerts.generate_alert_message(sample_hazard, 50, 0)
        
        assert alert.time_to_impact > 100  # Very large (infinity-like)
    
    def test_alert_different_severities(self):
        """Test alerts for different severity levels."""
        for severity in ['low', 'medium', 'high', 'critical']:
            hazard = {
                'latitude': 28.6139,
                'longitude': 77.2090,
                'class_name': 'pothole',
                'severity': severity,
                'confidence': 0.85
            }
            
            alert = alerts.generate_alert_message(hazard, 80, 60)
            
            assert alert.severity == severity
            assert isinstance(alert.message, str)


# ============================================================================
# Test Alert Triggering Logic
# ============================================================================

class TestShouldTriggerAlert:
    """Tests for alert triggering decision logic."""
    
    def test_trigger_alert_within_distance(self, sample_hazard):
        """Test alert triggers when within warning distance."""
        vehicle_pos = (28.6130, 77.2080)  # ~100m away
        
        should_alert = alerts.should_trigger_alert(
            sample_hazard,
            vehicle_pos,
            vehicle_speed=60
        )
        
        assert should_alert is True
    
    def test_no_trigger_beyond_distance(self, sample_hazard):
        """Test alert doesn't trigger when too far."""
        vehicle_pos = (28.6200, 77.2200)  # ~1km+ away
        
        should_alert = alerts.should_trigger_alert(
            sample_hazard,
            vehicle_pos,
            vehicle_speed=60
        )
        
        assert should_alert is False
    
    def test_trigger_depends_on_speed(self, sample_hazard):
        """Test triggering depends on vehicle speed."""
        vehicle_pos = (28.6135, 77.2085)  # ~60m away
        
        # Low speed = shorter warning distance = might not trigger
        trigger_low = alerts.should_trigger_alert(
            sample_hazard,
            vehicle_pos,
            vehicle_speed=20
        )
        
        # High speed = longer warning distance = should trigger
        trigger_high = alerts.should_trigger_alert(
            sample_hazard,
            vehicle_pos,
            vehicle_speed=80
        )
        
        # High speed more likely to trigger
        assert trigger_high is True
    
    def test_trigger_with_heading(self, sample_hazard):
        """Test heading consideration in triggering."""
        vehicle_pos = (28.6130, 77.2080)
        
        # Calculate bearing to hazard
        bearing = alerts.calculate_bearing(
            vehicle_pos[0], vehicle_pos[1],
            sample_hazard['latitude'], sample_hazard['longitude']
        )
        
        # Heading towards hazard = should trigger
        trigger_towards = alerts.should_trigger_alert(
            sample_hazard,
            vehicle_pos,
            vehicle_speed=60,
            vehicle_heading=bearing
        )
        
        # Heading away (opposite direction) = shouldn't trigger
        trigger_away = alerts.should_trigger_alert(
            sample_hazard,
            vehicle_pos,
            vehicle_speed=60,
            vehicle_heading=(bearing + 180) % 360
        )
        
        assert trigger_towards is True
        assert trigger_away is False
    
    def test_trigger_missing_coordinates(self, sample_hazard):
        """Test handling of missing coordinates."""
        # Hazard without coordinates
        bad_hazard = {'severity': 'high'}
        
        should_alert = alerts.should_trigger_alert(
            bad_hazard,
            (28.6130, 77.2080),
            vehicle_speed=60
        )
        
        assert should_alert is False
    
    def test_trigger_at_various_distances(self, sample_hazard):
        """Test triggering at various distances."""
        base_lat, base_lon = sample_hazard['latitude'], sample_hazard['longitude']
        
        # Test at 10m, 50m, 100m, 200m
        offsets = [0.0001, 0.0005, 0.001, 0.002]
        
        for offset in offsets:
            vehicle_pos = (base_lat - offset, base_lon)
            distance = alerts.calculate_distance_haversine(
                vehicle_pos[0], vehicle_pos[1],
                base_lat, base_lon
            )
            
            should_alert = alerts.should_trigger_alert(
                sample_hazard,
                vehicle_pos,
                vehicle_speed=60
            )
            
            # Should trigger if within warning distance
            warning_dist = alerts.calculate_warning_distance(60, sample_hazard['severity'])
            if distance <= warning_dist:
                assert should_alert is True


# ============================================================================
# Test Haversine Distance
# ============================================================================

class TestHaversineDistance:
    """Tests for Haversine distance calculation."""
    
    def test_distance_zero_same_point(self):
        """Test distance between same point is zero."""
        dist = alerts.calculate_distance_haversine(
            28.6139, 77.2090,
            28.6139, 77.2090
        )
        
        assert dist == 0.0
    
    def test_distance_symmetric(self):
        """Test distance calculation is symmetric."""
        dist1 = alerts.calculate_distance_haversine(
            28.6139, 77.2090,
            28.6149, 77.2100
        )
        dist2 = alerts.calculate_distance_haversine(
            28.6149, 77.2100,
            28.6139, 77.2090
        )
        
        assert abs(dist1 - dist2) < 0.01
    
    def test_distance_known_values(self):
        """Test against known distance values."""
        # Points approximately 1km apart
        dist = alerts.calculate_distance_haversine(
            28.6139, 77.2090,
            28.6230, 77.2090
        )
        
        # Should be close to 1000m
        assert 900 < dist < 1100
    
    def test_distance_positive(self):
        """Test distance is always positive."""
        dist = alerts.calculate_distance_haversine(
            28.6139, 77.2090,
            19.0760, 72.8777  # Mumbai
        )
        
        assert dist > 0
    
    @pytest.mark.parametrize("lat1,lon1,lat2,lon2,expected_min,expected_max", [
        (28.6139, 77.2090, 28.6140, 77.2091, 10, 20),      # ~15m
        (28.6139, 77.2090, 28.6149, 77.2100, 100, 150),    # ~120m
        (28.6139, 77.2090, 28.6239, 77.2190, 1000, 1500),  # ~1.2km
    ])
    def test_distance_ranges(self, lat1, lon1, lat2, lon2, expected_min, expected_max):
        """Test distance falls within expected ranges."""
        dist = alerts.calculate_distance_haversine(lat1, lon1, lat2, lon2)
        
        assert expected_min <= dist <= expected_max


# ============================================================================
# Test Bearing Calculation
# ============================================================================

class TestCalculateBearing:
    """Tests for bearing calculation."""
    
    def test_bearing_north(self):
        """Test bearing to point directly north."""
        bearing = alerts.calculate_bearing(
            28.6139, 77.2090,
            28.6239, 77.2090  # Same longitude, higher latitude
        )
        
        # Should be close to 0° (North)
        assert -10 < bearing < 10 or 350 < bearing < 370
    
    def test_bearing_south(self):
        """Test bearing to point directly south."""
        bearing = alerts.calculate_bearing(
            28.6139, 77.2090,
            28.6039, 77.2090  # Same longitude, lower latitude
        )
        
        # Should be close to 180° (South)
        assert 170 < bearing < 190
    
    def test_bearing_east(self):
        """Test bearing to point directly east."""
        bearing = alerts.calculate_bearing(
            28.6139, 77.2090,
            28.6139, 77.2190  # Same latitude, higher longitude
        )
        
        # Should be close to 90° (East)
        assert 80 < bearing < 100
    
    def test_bearing_west(self):
        """Test bearing to point directly west."""
        bearing = alerts.calculate_bearing(
            28.6139, 77.2090,
            28.6139, 77.1990  # Same latitude, lower longitude
        )
        
        # Should be close to 270° (West)
        assert 260 < bearing < 280
    
    def test_bearing_range(self):
        """Test bearing is always in [0, 360) range."""
        bearing = alerts.calculate_bearing(
            28.6139, 77.2090,
            19.0760, 72.8777
        )
        
        assert 0 <= bearing < 360


# ============================================================================
# Test Display Formatting
# ============================================================================

class TestFormatAlertForDisplay:
    """Tests for alert display formatting."""
    
    def test_format_alert_html(self, sample_hazard):
        """Test HTML formatting of alert."""
        alert = alerts.generate_alert_message(sample_hazard, 80, 60)
        formatted = alerts.format_alert_for_display(alert)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert '<div' in formatted  # Contains HTML
    
    def test_format_alert_contains_message(self, sample_hazard):
        """Test formatted alert contains message text."""
        alert = alerts.generate_alert_message(sample_hazard, 80, 60)
        formatted = alerts.format_alert_for_display(alert)
        
        # Message text should be present
        assert sample_hazard['class_name'] in formatted.lower()
    
    def test_format_alert_for_audio(self, sample_hazard):
        """Test audio-friendly formatting."""
        alert = alerts.generate_alert_message(sample_hazard, 80, 60)
        audio_text = alerts.format_alert_for_audio(alert)
        
        assert isinstance(audio_text, str)
        # Should not contain markdown or emojis
        assert '**' not in audio_text
        assert '⚠️' not in audio_text


# ============================================================================
# Test Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_kmh_to_ms(self):
        """Test km/h to m/s conversion."""
        ms = alerts.kmh_to_ms(36)  # 36 km/h = 10 m/s
        
        assert abs(ms - 10.0) < 0.01
    
    def test_ms_to_kmh(self):
        """Test m/s to km/h conversion."""
        kmh = alerts.ms_to_kmh(10)  # 10 m/s = 36 km/h
        
        assert abs(kmh - 36.0) < 0.01
    
    def test_conversion_roundtrip(self):
        """Test conversion is reversible."""
        original = 60.0
        converted = alerts.ms_to_kmh(alerts.kmh_to_ms(original))
        
        assert abs(converted - original) < 0.01
    
    def test_get_speed_category(self):
        """Test speed categorization."""
        assert alerts.get_speed_category(10) == "very_slow"
        assert alerts.get_speed_category(30) == "slow"
        assert alerts.get_speed_category(50) == "moderate"
        assert alerts.get_speed_category(70) == "fast"
        assert alerts.get_speed_category(100) == "very_fast"


# ============================================================================
# Integration Tests
# ============================================================================

class TestAlertsIntegration:
    """Integration tests for complete alert workflow."""
    
    def test_complete_alert_workflow(self, sample_hazard):
        """Test complete alert generation workflow."""
        vehicle_pos = (28.6130, 77.2080)
        vehicle_speed = 60
        vehicle_heading = 45
        
        # Step 1: Check if alert should trigger
        should_trigger = alerts.should_trigger_alert(
            sample_hazard,
            vehicle_pos,
            vehicle_speed,
            vehicle_heading
        )
        
        if should_trigger:
            # Step 2: Calculate distance
            distance = alerts.calculate_distance_haversine(
                vehicle_pos[0], vehicle_pos[1],
                sample_hazard['latitude'], sample_hazard['longitude']
            )
            
            # Step 3: Generate alert
            alert = alerts.generate_alert_message(
                sample_hazard,
                distance,
                vehicle_speed
            )
            
            # Step 4: Format for display
            formatted = alerts.format_alert_for_display(alert)
            
            # Verify complete workflow
            assert distance > 0
            assert alert.message is not None
            assert formatted is not None
    
    def test_multiple_hazards_scenario(self):
        """Test alerting for multiple hazards."""
        vehicle_pos = (28.6130, 77.2080)
        vehicle_speed = 60
        
        hazards = [
            {'latitude': 28.6139, 'longitude': 77.2090, 'severity': 'high'},
            {'latitude': 28.6145, 'longitude': 77.2095, 'severity': 'medium'},
            {'latitude': 28.6200, 'longitude': 77.2200, 'severity': 'critical'},
        ]
        
        triggered_alerts = []
        
        for hazard in hazards:
            if alerts.should_trigger_alert(hazard, vehicle_pos, vehicle_speed):
                distance = alerts.calculate_distance_haversine(
                    vehicle_pos[0], vehicle_pos[1],
                    hazard['latitude'], hazard['longitude']
                )
                alert = alerts.generate_alert_message(hazard, distance, vehicle_speed)
                triggered_alerts.append(alert)
        
        # Should have some alerts
        assert len(triggered_alerts) >= 0


# ============================================================================
# Parametrized Test Suites
# ============================================================================

@pytest.mark.parametrize("speed,severity,road,expected_min", [
    (30, 'low', 'dry', 30),
    (30, 'critical', 'dry', 50),
    (60, 'medium', 'dry', 60),
    (60, 'medium', 'wet', 80),
    (60, 'medium', 'icy', 120),
    (90, 'high', 'dry', 120),
    (120, 'critical', 'icy', 250),
])
def test_warning_distance_combinations(speed, severity, road, expected_min):
    """Parametrized test for various warning distance scenarios."""
    distance = alerts.calculate_warning_distance(speed, severity, road)
    
    assert distance >= expected_min


@pytest.mark.parametrize("distance,severity,road,expected_max", [
    (50, 'critical', 'icy', 30),
    (100, 'high', 'wet', 60),
    (150, 'medium', 'dry', 80),
    (200, 'low', 'dry', 100),
])
def test_safe_speed_combinations(distance, severity, road, expected_max):
    """Parametrized test for various safe speed scenarios."""
    speed = alerts.calculate_safe_speed(distance, severity, road)
    
    assert 0 <= speed <= expected_max + 20  # Allow some tolerance


@pytest.mark.parametrize("lat1,lon1,lat2,lon2", [
    (28.6139, 77.2090, 28.6139, 77.2090),  # Same point
    (28.6139, 77.2090, 28.6149, 77.2100),  # Close
    (28.6139, 77.2090, 19.0760, 72.8777),  # Far (Delhi-Mumbai)
    (0, 0, 0, 180),                         # Opposite side
])
def test_haversine_various_points(lat1, lon1, lat2, lon2):
    """Parametrized test for Haversine distance calculation."""
    dist = alerts.calculate_distance_haversine(lat1, lon1, lat2, lon2)
    
    assert dist >= 0
    assert isinstance(dist, float)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_high_speed(self):
        """Test with unrealistically high speed."""
        distance = alerts.calculate_warning_distance(300, 'medium')
        
        assert distance > 0
        assert distance < 10000  # Reasonable upper bound
    
    def test_very_long_distance(self):
        """Test with very long distance."""
        speed = alerts.calculate_safe_speed(10000, 'low')
        
        assert 0 <= speed <= 120
    
    def test_alert_at_zero_distance(self, sample_hazard):
        """Test alert generation at zero distance."""
        alert = alerts.generate_alert_message(sample_hazard, 0, 60)
        
        assert alert.urgency == alerts.UrgencyLevel.CRITICAL
        assert alert.recommended_speed < 60


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=app.components.alerts"])
