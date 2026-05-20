#!/usr/bin/env python3

import csv
import math
import itertools


CSV_FILE = "wheel_data.csv"

WAYPOINTS = [
    (0.4, 0.0),
    (0.6, 0.346),
    (0.4, 0.692),
    (0.0, 0.692),
    (-0.2, 0.346),
    (0.0, 0.0),
    (0.2, 0.0),
    (0.2, -0.4),
]


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def load_data(filename):
    rows = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "t": float(row["t"]),
                "set_L": float(row["set_L"]),
                "set_R": float(row["set_R"]),
                "vel_L": float(row["vel_L"]),
                "vel_R": float(row["vel_R"]),
            })
    return rows


def estimate_gain(rows):
    ratios_L = []
    ratios_R = []

    for r in rows:
        if abs(r["set_L"]) > 0.1 and abs(r["vel_L"]) > 0.05:
            ratios_L.append(r["vel_L"] / r["set_L"])

        if abs(r["set_R"]) > 0.1 and abs(r["vel_R"]) > 0.05:
            ratios_R.append(r["vel_R"] / r["set_R"])

    gain_L = sum(ratios_L) / len(ratios_L)
    gain_R = sum(ratios_R) / len(ratios_R)

    return gain_L, gain_R


def simulate(kp_dist, ki_dist, kd_dist, kp_ang, ki_ang, kd_ang, gain_L, gain_R):
    x = 0.0
    y = 0.0
    theta = 0.0

    r = 0.05
    b = 0.19
    dt = 0.02

    v_max = 0.08
    w_max = 0.35
    pos_tol = 0.05

    int_dist = 0.0
    int_ang = 0.0
    prev_dist = 0.0
    prev_ang = 0.0

    score = 0.0
    total_time = 0.0

    for gx, gy in WAYPOINTS:
        reached = False

        for _ in range(1500):
            dx = gx - x
            dy = gy - y

            dist = math.sqrt(dx * dx + dy * dy)
            desired_theta = math.atan2(dy, dx)
            ang = wrap_angle(desired_theta - theta)

            if dist < pos_tol:
                reached = True
                break

            int_dist += dist * dt
            int_ang += ang * dt

            int_dist = max(-1.0, min(int_dist, 1.0))
            int_ang = max(-1.0, min(int_ang, 1.0))

            der_dist = (dist - prev_dist) / dt
            der_ang = (ang - prev_ang) / dt

            v_cmd = kp_dist * dist + ki_dist * int_dist + kd_dist * der_dist
            w_cmd = kp_ang * ang + ki_ang * int_ang + kd_ang * der_ang

            v_cmd = max(-v_max, min(v_cmd, v_max))
            w_cmd = max(-w_max, min(w_cmd, w_max))

            if abs(ang) > 0.5:
                v_cmd = 0.0

            v_left = v_cmd - w_cmd * b / 2.0
            v_right = v_cmd + w_cmd * b / 2.0

            set_L = v_left / r
            set_R = v_right / r

            real_wL = gain_L * set_L
            real_wR = gain_R * set_R

            real_vL = real_wL * r
            real_vR = real_wR * r

            real_v = (real_vR + real_vL) / 2.0
            real_w = (real_vR - real_vL) / b

            x += real_v * math.cos(theta) * dt
            y += real_v * math.sin(theta) * dt
            theta += real_w * dt
            theta = wrap_angle(theta)

            score += dist + 0.15 * abs(ang)
            score += 0.05 * abs(w_cmd)
            total_time += dt

            prev_dist = dist
            prev_ang = ang

        if not reached:
            score += 100.0

    score += 0.05 * total_time
    return score


def main():
    rows = load_data(CSV_FILE)
    gain_L, gain_R = estimate_gain(rows)

    print("\nModelo estimado desde datos reales")
    print("----------------------------------")
    print(f"gain_L = {gain_L:.3f}")
    print(f"gain_R = {gain_R:.3f}")

    kp_dist_values = [0.25, 0.35, 0.45, 0.55, 0.65,0.75,0.85,0.95,1.0,2.0,3.0,4.0,5.0,6.0]
    ki_dist_values = [0.0, 0.005, 0.01,0.03,0.1,0.2,0.3,0.09,0.05,1.0]
    kd_dist_values = [0.0,0.25, 0.35, 0.45, 0.55, 0.65,0.75,0.85,0.95,1.0]

    kp_ang_values = [1.0, 1.3, 1.6, 1.9, 2.2,3.2,0.8,0.75,0.05,0.01]
    ki_ang_values = [0.0,0.0001,0.0002,0.001,0.0004,0.05]
    kd_ang_values = [0.0, 0.01, 0.02,0.0001,1.0,2.0,0.1,0.2,0.35]

    best = None

    for kp_dist, ki_dist, kd_dist, kp_ang, ki_ang, kd_ang in itertools.product(
        kp_dist_values,
        ki_dist_values,
        kd_dist_values,
        kp_ang_values,
        ki_ang_values,
        kd_ang_values,
    ):
        score = simulate(
            kp_dist,
            ki_dist,
            kd_dist,
            kp_ang,
            ki_ang,
            kd_ang,
            gain_L,
            gain_R,
        )

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "kp_dist": kp_dist,
                "ki_dist": ki_dist,
                "kd_dist": kd_dist,
                "kp_ang": kp_ang,
                "ki_ang": ki_ang,
                "kd_ang": kd_ang,
            }

    print("\nMejores ganancias estimadas")
    print("---------------------------")
    print(f"score   = {best['score']:.3f}")
    print(f"kp_dist = {best['kp_dist']}")
    print(f"ki_dist = {best['ki_dist']}")
    print(f"kd_dist = {best['kd_dist']}")
    print(f"kp_ang  = {best['kp_ang']}")
    print(f"ki_ang  = {best['ki_ang']}")
    print(f"kd_ang  = {best['kd_ang']}")

    print("\nComando para probar")
    print("-------------------")
    print(
        "ros2 run waypoints_puzzlebot waypoints --ros-args "
        f"-p kp_dist:={best['kp_dist']} "
        f"-p ki_dist:={best['ki_dist']} "
        f"-p kd_dist:={best['kd_dist']} "
        f"-p kp_ang:={best['kp_ang']} "
        f"-p ki_ang:={best['ki_ang']} "
        f"-p kd_ang:={best['kd_ang']} "
        "-p v_max:=0.08 "
        "-p omega_max:=0.35 "
        "-p pos_tol:=0.05"
    )


if __name__ == "__main__":
    main()
