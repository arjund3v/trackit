def normalize_angle(angle):
    return (angle + 180) % 360 - 180

def unwrap_angle(current, previous):
    candidates = [
        current - 360,
        current - 180,
        current,
        current + 180,
        current + 360
    ]
    return min(candidates, key=lambda angle: abs(angle - previous))

def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0.0

    return value