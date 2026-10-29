from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RayTrajectory:
    """A light ray of beam :3 """

    phi: FloatArray
    
    u: FloatArray # convineit number 

    @property
    def radius(self) -> FloatArray:
        """converts radius into ---> Schwarzschild radius"""
        radius = np.full_like(self.u, np.inf)
        
        np.divide(1.0, self.u, out=radius, where=self.u > 0)
        
        return radius

    @property
    def deflection_radian(self) ->float | None: # convering phi or angle to deflection radian
        if not self.escaped or self.escape_angle is None:
            return None
        return 2.0 * self.escape_angle - np.pi
    
    @property
    def deflection_angle(self)-> float | None:  #radian --> theta (degree)
        angle=self.deflection_radian
        return None if angle is None else float(np.degrees(angle))
    

def _rhs(
    phi: float,
    state: FloatArray,
) -> tuple[float, float]:
    
    """First-order Schwarzschild photon equation (diff eq !)"""
    
    del phi

    u, du_dphi = state
    d2u_dphi2 = 1.5 * u**2 - u

    return du_dphi, d2u_dphi2

def _escape_event(phi, state):
    
    u , du_dphi=state

    return u

_escape_event.terminal = True
_escape_event.direction = -1





def simulate_ray(
    closest_approach: float = 10.0,
    *,
    max_angle: float = 8.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> RayTrajectory:
    """Integrate a scattering ray from its point of closest approach """
    
    if not np.isfinite(closest_approach) or closest_approach <= 1.5:
        raise ValueError("closest_approach must be finite and > 1.5.")

    
    if not np.isfinite(max_angle) or max_angle <= 0:
        raise ValueError("max_angle must be finite and positive.")

    
    if not (0 < rtol < 1 and 0 < atol < 1):
        raise ValueError("rtol and atol must be between 0 and 1.")

    u0 = 1.0 / closest_approach

    solution = solve_ivp(
        _rhs,
        (0.0, max_angle),
        y0=(u0, 0.0),  # Initial conditions
        method="DOP853",
        rtol=rtol,
        atol=atol,
        events=_escape_event,
        max_step=0.02,
    )

    if not solution.success:
        raise RuntimeError(solution.message)
    print("Escape angle:", float(((2*solution.t[-1])-np.pi)*(57.2958)))
    print("Final u:", solution.y[0, -1])
    print("Event detected:", len(solution.t_events[0]) > 0)


    # if phi=0 sym eq
    phi_out = solution.t
    u_out = solution.y[0]

    phi = np.concatenate((-phi_out[:0:-1], phi_out))
    u = np.concatenate((u_out[:0:-1], u_out))

    return RayTrajectory(phi=phi, u=u)


closest_approaches = [3.0, 5.0, 10.0]
for distance in closest_approaches:
    trajectory = simulate_ray(closest_approach=distance)

    print(
        f"Closest approach: {distance} rs | "
        f"Number of points: {len(trajectory.phi)}"

    )
    print("----------------------------------------")