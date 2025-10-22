import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AIR_DENSITY = 1.2   # [kg/m³]
FLAP_LENGTH = 0.12  # [m]


def get_R_ff_to_bf(theta, plate_num):
    """
    Returns the rotation matrix R_ff_to_bf and radial unit vector
    for the given plate number. Each plate has a distinct orientation
    and position offset relative to the base frame.

    Inputs
    ----------
    theta : float
        Rotation angle of flap in radians.
    plate_num : int
        Plate number (1-4).

    Returns
    -------
    R_ff_to_bf : np.ndarray
        3x3 rotation matrix (rotation due to flap actuation).
    r : np.ndarray
        3x1 unit vector (radial direction) --> useful to get p_ff_wrt_bf and the direction of omega_flap for every flap.
    """
    c, s = np.cos(theta), np.sin(theta)

    match plate_num:
        case 1:
            R = np.array([
                [1, 0,  0],
                [0, c, -s],
                [0, s,  c]
            ])
            r = np.array([1.0, 0.0, 0.0])

        case 2:
            R = np.array([
                [ c, 0, s],
                [ 0, 1, 0],
                [-s, 0, c]
            ])
            r = np.array([0.0, 1.0, 0.0])

        case 3: # Same of case 1 but x and y mirrored due to opposed position on drone
            R = np.array([
                [-1, 0,  0],
                [ 0, -c, s],
                [ 0, s,  c]
            ])
            r = np.array([-1.0, 0.0, 0.0])

        case 4: # Same of case 2 but x and y mirrored due to opposed position on drone
            R = np.array([
                [-c, 0,  -s],
                [ 0, -1, 0],
                [-s, 0,  c]
            ])
            r = np.array([0.0, -1.0, 0.0])

        case _:
            raise ValueError("Plate number is not valid (must be 1-4)")

    return R, r
        


def R_from_rpy(rpy_angles):
    """
    Compute rotation matrix between base frame and WRF using: Roll (phi), Pitch (psi), Yaw (gamma) angles
    R = Rz(gamma) * Ry(psi) * Rx(phi)

    Input
    ----------
    rpy_angles : array-like of length 3
        [phi, psi, gamma] in radians
        (roll, pitch, yaw)

    Returns
    -------
    R : (3, 3) numpy.ndarray
        Rotation matrix expressing bf w.r.t. WRF
    """
    phi, psi, gamma = rpy_angles  # unpack angles

    cphi, sphi = np.cos(phi), np.sin(phi)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    cgamma, sgamma = np.cos(gamma), np.sin(gamma)

    R = np.array([
        [cgamma*cpsi,  cgamma*spsi*sphi - sgamma*cphi,  cgamma*spsi*cphi + sgamma*sphi],
        [sgamma*cpsi,  sgamma*spsi*sphi + cgamma*cphi,  sgamma*spsi*cphi - cgamma*sphi],
        [-spsi,        cpsi*sphi,                       cpsi*cphi]
    ])

    return R


def compute_drag_force (rho, A, cd , v_flap_WRF, v_air_WRF, n_WRF):
    '''
    Computation of drag force taking as input geometric/physical parameters:
    air density, area of the flap, drag coefficient, normal direction of flap in WRF
    And flap kinematics:
    velocity of airflow and of flap in WRF and theta.
    Theta is the sum of flap angle wrt base and angle of the base wrt WRF if the model is planar

    '''

    # Computation of velocity difference and projection perpendicular to flap
    v_rel = v_air_WRF - v_flap_WRF
    v_perp_mag = np.dot(v_rel, n_WRF)

    # Aerodynamic drag force magnitude
    F_mag = 0.5 * rho * cd * A * (v_perp_mag ** 2)

    # Projection along flap and sign adaptation
    F_vec =  F_mag * n_WRF * np.sign(v_perp_mag) 

    return F_vec



# example of plates dictionary:
plates = [
    {"num": 1, "cd": 1.0, "area": 0.2, "length" : FLAP_LENGTH, "cop_frf": [0.0, 0.0, 0.0], "theta_frf" : np.deg2rad(25), "theta_dot_frf" : 0},
    {"num": 2, "cd": 1.0, "area": 0.2, "length" : FLAP_LENGTH, "cop_frf": [0.0, 0.0, 0.0], "theta_frf" : np.deg2rad(25), "theta_dot_frf" : 0},
    {"num": 3, "cd": 1.0, "area": 0.2, "length" : FLAP_LENGTH, "cop_frf": [0.0, 0.0, 0.0], "theta_frf" : np.deg2rad(-25), "theta_dot_frf" : 0},
    {"num": 4, "cd": 1.0, "area": 0.2, "length" : FLAP_LENGTH, "cop_frf": [0.0, 0.0, 0.0], "theta_frf" : np.deg2rad(-25), "theta_dot_frf" : 0},
]

def drag_forces_torques(plates, v_air_WRF, v_com_WRF, rpy_angles_WRF, rpy_rates_WRF, rho = AIR_DENSITY):
    """
    Compute the drag force and associated torques for each plate and sum them.

    Parameters
    ----------
    plates : list of dict
        Each dict must contain
            - num : int (plate number)
            - cd : float   (drag coefficient)
            - area : float (area of the flap, m^2)
            - length : float (distance between CoM and flap reference frame) --> here I now place the flap frame at the tip of the flap
            - cop : list or tuple (center of pressure in flap reference frame)

    
    v_air_WRF : 3x1 airflow velocity (m/s) in WRF
    v_com_WRF : 3x1 Floaty CoM velocity (m/s) in WRF
    rpy_angles_WRF : 3x1 Roll, Pitch, Yaw angles of floaty (rad) in WRF
    rpy_rates_WRF : 3x1 Roll, Pitch, Yaw velocities of floaty (rad/s) in WRF

    rho (default) : float (air density, kg/m^3)
    Returns
    -------
    F : 3x1 Total Force vector in WRF (N)
    T : 3x1 Total Torque vector in WRF (Nm)

    """
    # Just for safety
    v_air_WRF = np.asarray(v_air_WRF, dtype=float)
    v_com_WRF = np.asarray(v_com_WRF, dtype=float)
    rpy_angles_WRF = np.asarray(rpy_angles_WRF, dtype=float)
    rpy_rates_WRF = np.asarray(rpy_rates_WRF, dtype=float)

    # Force and torque accumulation vectors
    F = np.array([0.0, 0.0, 0.0])
    T = np.array([0.0, 0.0, 0.0])

    # Rotation matrix between Floaty base and WRF
    R_bf_to_WRF = R_from_rpy(rpy_angles_WRF)

    # Loop over the plates
    results = []
    for plate in plates:

        plate_num = plate["num"]
        A = plate["area"]
        cd = plate["cd"]
        flap_length = plate["length"]
        cop_ff = np.asarray(plate["cop_frf"], dtype=float)
        theta_flap = plate["theta_frf"]
        theta_dot_ff = plate["theta_dot_frf"]

        # Get rotation of plate due to actuation expressed in base frame + get radial oriented vector (for omeega_flap direction)
        R_ff_to_bf, r_plate = get_R_ff_to_bf(theta_flap, plate_num)

        # Rotation of the plate expressed in WRF
        R_ff_to_WRF = R_bf_to_WRF @ R_ff_to_bf

        # Normal vector of the plate in WRF
        n_WRF = R_ff_to_WRF[:,2]

        # Center of Pressure coordinates wrt floaty base expressed in WRF (can be a separate function in case...)
        p_ff_wrt_bf = flap_length * r_plate
        cop_bf = R_ff_to_bf @ cop_ff + p_ff_wrt_bf
        cop_WRF = R_bf_to_WRF @ cop_bf

        # Vector (oriented) angular velocity of flap in flap reference frame
        omega_flap_ff = theta_dot_ff * r_plate

        # Vector (oriented) angular velocity of flap in WRF
        omega_flap_WRF = R_ff_to_WRF @ omega_flap_ff

        # Lever arm of the flap in WRF
        flap_lever_arm_WRF = R_ff_to_WRF @ cop_ff

        # Rigid body velocity of flap as sum of Vcom + omega_floaty_rpy x (CoP_WRF - CoM_WRF) + omega_flap_WRF x flap_lever_arm_WRF
        v_flap_WRF = v_com_WRF + np.cross(rpy_rates_WRF, cop_WRF) + np.cross(omega_flap_WRF, flap_lever_arm_WRF)

        # Update force and torque vectors
        F_flap = compute_drag_force (rho, A, cd , v_flap_WRF, v_air_WRF, n_WRF)
        T_flap = np.cross(cop_WRF,F_flap)

        F += F_flap
        T += T_flap

        results.append({
            "plate": plate_num,
            "F_x": F_flap[0],
            "F_y": F_flap[1],
            "F_z": F_flap[2],
            "T_x": T_flap[0],
            "T_y": T_flap[1],
            "T_z": T_flap[2]
        })

    return F,T, results


F, T, results = drag_forces_torques(plates, [0,0,10], [0,0,0], [0,0,0], [0,0,0], rho=AIR_DENSITY)

df = pd.DataFrame(results).set_index("plate")

print("\nPer-flap forces and torques:")
print(df.round(5).to_string())

print("\nTotal Force:", F)
print("Total Torque:", T)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Force plot
df[["F_x", "F_y", "F_z"]].plot(kind="bar", ax=axes[0], title="Per-flap Forces (N)")
axes[0].axhline(0, color="k", linewidth=0.8)
axes[0].set_xlabel("Plate")
axes[0].legend(loc="upper right")

# Torque plot
df[["T_x", "T_y", "T_z"]].plot(kind="bar", ax=axes[1], title="Per-flap Torques (Nm)")
axes[1].axhline(0, color="k", linewidth=0.8)
axes[1].set_xlabel("Plate")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()
