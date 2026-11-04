import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab.geodesics import simulate_ray


def impact_parameter(r0):
    return r0 / np.sqrt(1.0 - 1.0 / r0)


def main():
    rows = []
    for r0 in [10.0, 15.0, 20.0, 30.0, 50.0, 80.0]:
        ray = simulate_ray(closest_approach=r0, max_angle=8.0)
        b_bar = impact_parameter(r0)
        weak = 2.0 / b_bar

        if ray.escaped and ray.deflection_radian is not None:
            numerical = ray.deflection_radian
            relative_error = abs(numerical - weak) / abs(numerical)
            status = "escaped"
        else:
            numerical = ""
            relative_error = ""
            status = "unresolved"

        rows.append({
            "r_min_rs": r0,
            "impact_parameter_rs": b_bar,
            "numerical_rad": numerical,
            "weak_field_rad": weak,
            "relative_error": relative_error,
            "status": status,
        })

    output = Path("outputs/weak_field.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if row["status"] == "escaped"]
    plt.loglog(
        [row["impact_parameter_rs"] for row in valid],
        [row["numerical_rad"] for row in valid],
        "o-",
        label="DOP853",
    )
    plt.loglog(
        [row["impact_parameter_rs"] for row in valid],
        [row["weak_field_rad"] for row in valid],
        "--",
        label="Leading weak-field approximation",
    )
    plt.xlabel("Impact parameter b / r_s")
    plt.ylabel("Deflection (radians)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/weak_field.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()