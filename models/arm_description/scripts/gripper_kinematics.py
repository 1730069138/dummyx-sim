#!/usr/bin/env python3
"""Kinematic coupling only. Inputs radians, jaw positions metres. No hardware I/O."""
import math

LOWER = -1.999307717981279
UPPER = 0.0
RADIUS = 0.03
ROD_LENGTH = 0.04000000701249939
PIN_Z = -0.015
CLOSED_SLIDER_X = 0.01525
CLOSED_CRANK = 0.9671723727474626

def coupled_positions(angle):
    if not math.isfinite(angle) or not LOWER-1e-9 <= angle <= UPPER+1e-9:
        raise ValueError('joint7 outside calibrated closed-to-open interval')
    theta = CLOSED_CRANK + angle
    ax = -RADIUS * math.sin(theta)
    az = -RADIUS * math.cos(theta)
    slider = ax + math.sqrt(ROD_LENGTH**2 - (PIN_Z-az)**2)
    alpha = -math.atan2(PIN_Z-az, slider-ax)
    alpha0 = -math.atan2(PIN_Z+RADIUS*math.cos(CLOSED_CRANK), CLOSED_SLIDER_X+RADIUS*math.sin(CLOSED_CRANK))
    rod_angle = (alpha-alpha0+math.pi) % (2*math.pi) - math.pi
    opening = -(slider-CLOSED_SLIDER_X)
    return {'joint10':opening, 'joint12':opening, 'joint11':rod_angle, 'joint13':rod_angle}

if __name__ == '__main__':
    import argparse, json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('angle_deg',type=float)
    args=p.parse_args()
    print(json.dumps(coupled_positions(math.radians(args.angle_deg)),indent=2))
