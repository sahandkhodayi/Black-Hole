from lab.geodesics import simulate_ray
from lab.plot import plot_trajectory


def main() -> None:
    ray = simulate_ray(closest_approach=10.0)

    plot_trajectory(ray, "outputs/first_ray.png")

    print(f"Trajectory points: {len(ray.phi)}")
    print(f"Closest radius: {ray.radius.min():.6f} r_s")
    print("Plot saved to outputs/first_ray.png")


if __name__ == "__main__":
    main()