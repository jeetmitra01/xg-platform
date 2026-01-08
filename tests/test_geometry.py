from src.features.geometry import shot_distance, shot_angle

def test_distance_positive():
    d = shot_distance(100, 40)
    assert d > 0

def test_angle_range():
    a = shot_angle(100, 40)
    assert 0 <= a <= 3.14159
