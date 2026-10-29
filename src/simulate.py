from lab.geodesics import simulate_ray
from lab.plot import plot_rays


def main() -> None:
    ray = simulate_ray(closest_approach=5.0)

    
    print("===========================================")
    print("u min:", ray.u.min())
    print("u max:", ray.u.max())
    print("radius min:", ray.radius.min())
    print("radius max:", ray.radius.max())
    print("number of points:", len(ray.phi))
    print(f"Trajectory points: {len(ray.phi)}")
    print(f"Closest radius: {ray.radius.min():.6f} r_s")
    print("Plot saved to outputs/first_ray.png")
    plot_rays(closest_approaches=(3.0, 5.0, 10.0))


if __name__ == "__main__":
    main()