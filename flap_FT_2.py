import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# I need to:
# - switch to new flap frames
# - insert an option for base reference frame rotated of 45 degs wrt Z_base
# - insert mass for every plate
# - insert some base plates


AIR_DENSITY = 1.2   # [kg/m³]
FLAP_OFFSET = 0.12  # [m]

#########################################
# Helper functions for the 3D model

def get_position (flap_num, flap_offset = FLAP_OFFSET):
    """
    Returns the rotation matrix R_ff_to_bf and radial unit vector
    for the given plate number. Each plate has a distinct orientation
    and position offset relative to the base frame.

    Inputs
    ----------
    flap_num : int
        flap number (1-4) referring to the actuation joint (many straights and kinks share same flap number)

    Returns
    -------
    pf_wrt_bf : 3x1 postion vector wrt base frame
        position of flap frame wrt base frame
    """

    match flap_num:
        case 1:
            pf_wrt_bf = [flap_offset,0.0,0.0]
        case 2:
            pf_wrt_bf = [0.0,flap_offset,0.0]

        case 3: # Same of case 1 but mirrored due to opposed position on drone
            pf_wrt_bf = [-flap_offset,0.0,0.0]

        case 4: # Same of flap 2 but mirrored due to opposed position on drone
            pf_wrt_bf = [0.0,-flap_offset,0.0]

        case _:
            raise ValueError("Plate number is not valid (must be 1-4) - if you added flaps/base plates update this function")

    return pf_wrt_bf
        

def get_orientation(flap_number, theta_plate, kink_pitch = None, kink_yaw = None): # verified, it works
# Still not taking into account the base plates...
    '''
    Takes as input:

    -   flap_number: the number of the flap (which gives the placement of the plate relatively to the base and is common between
            straights and kinks belonging to same joint
    -   theta_plate: the actuation angle
    -   kink_pitch: (None if it is a straight element) the pitch angle between straight flap and the kink
    -   kink_yaw: (None if it is a straight element) the yaw angle between straight flap and the kink

    '''
    # 90 degs rotation abut z
    switch_flap =  np.array([
        [0,-1,0],
        [1,0,0],
        [0,0,1]
    ])

    # In this new Plate Reference Frame, all plates rotate about x_plate axis, this is the first flap's rotation matrix
    R1 =  np.array([
            [1,       0,                    0          ],
            [0,np.cos(theta_plate),-np.sin(theta_plate)],
            [0,np.sin(theta_plate), np.cos(theta_plate)]
        ])

    # The rotation matrix of i-th flap can be computed like this (every flap frame is the previous but rotated by 90 degs about z):
    R = np.linalg.matrix_power(switch_flap, flap_number-1) @ R1

    # Tell straights and kinks apart:

    if (kink_pitch == None) and (kink_yaw == None): 
        return R

    if (kink_pitch != None) and (kink_yaw != None):

        # For the kinks we need to add also the pitch and yaw angle offsets:
        R_pitch =  np.array([
                [1,       0,                   0         ],
                [0,np.cos(kink_pitch),-np.sin(kink_pitch)],
                [0,np.sin(kink_pitch), np.cos(kink_pitch)]
            ])

        R_yaw =  np.array([
                [np.cos(kink_yaw),-np.sin(kink_yaw),  0],
                [np.sin(kink_yaw), np.cos(kink_yaw),  0],
                [       0,                 0,         1]
            ])

        return R_pitch @ R_yaw @ R
    
    else: print("Insert a valid orientation for plate reference frame wrt base frame")


#print(get_orientation(4,np.deg2rad(30), np.deg2rad(0),np.deg2rad(0))) # sanity check for orientation function

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

#############################################

# example of plates dictionary: (torque generated by plate weight is neglected so we don't need CoM position)
plates = [
    {"name": "Straight 1", "cd": 1.0, "mass" : 0.01, "area": 0.2, "plate_frame_pos" : get_position(1),
     "plate_frame_orientation" :  get_orientation(1,np.deg2rad(25)), "cop_prf": [0.0, 0.0, 0.0],
     "plate_lin_vel_WRF": [0.0, 0.0, 0.0], "airflow_vel_WRF": [0.0, 0.0, 10.0]},

    {"name": "Straight 2", "cd": 1.0, "mass" : 0.01, "area": 0.2, "plate_frame_pos" : get_position(2),
     "plate_frame_orientation" :  get_orientation(1,np.deg2rad(25)), "cop_prf": [0.0, 0.0, 0.0],
     "plate_lin_vel_WRF": [0.0, 0.0, 0.0], "airflow_vel_WRF": [0.0, 0.0, 10.0]},

    {"name": "Straight 3", "cd": 1.0, "mass" : 0.01, "area": 0.2, "plate_frame_pos" : get_position(3),
     "plate_frame_orientation" :  get_orientation(1,np.deg2rad(25)), "cop_prf": [0.0, 0.0, 0.0],
     "plate_lin_vel_WRF": [0.0, 0.0, 0.0], "airflow_vel_WRF": [0.0, 0.0, 10.0]},

    {"name": "Straight 4", "cd": 1.0, "mass" : 0.01, "area": 0.2, "plate_frame_pos" : get_position(4),
     "plate_frame_orientation" :  get_orientation(1,np.deg2rad(25)), "cop_prf": [0.0, 0.0, 0.0],
     "plate_lin_vel_WRF": [0.0, 0.0, 0.0], "airflow_vel_WRF": [0.0, 0.0, 10.0]}
    ]

def drag_forces_torques(plates, v_com_WRF, rpy_angles_WRF, rpy_rates_WRF, rho = AIR_DENSITY, g = 9.81):
    """
    Compute the drag force and associated torques for each plate and sum them.

    Parameters
    ----------
    plates : list of dict
        Each dict must contain
            - name  : string (plate identification)
            - mass  : float (mass of the plate)
            - cd    : float   (drag coefficient)
            - area  : float (area of the flap, m^2)
            - plate_frame_pos           : 3x1 array containing the origin of plate RF in base RF
            - plate_frame_orientation   : 3x3 matrix containing the orientation (rotation matrix) of plate RF in base RF
            - cop_prf                   : 3x1 array containing center of pressure in flap reference frame

    
    v_com_WRF       : 3x1 Floaty CoM velocity (m/s) in WRF
    rpy_angles_WRF  : 3x1 Roll, Pitch, Yaw angles of floaty (rad) in WRF
    rpy_rates_WRF   : 3x1 Roll, Pitch, Yaw velocities of floaty (rad/s) in WRF

    rho (default)   : float (air density, kg/m^3)
    g (default)     : float (gravity acceleration, m/s^2)

    Returns
    -------
    F : 3x1 Total Force vector in WRF (N)
    T : 3x1 Total Torque vector in WRF (Nm)

    """
    # Just for safety
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

        plate_name = plate["name"]
        m = plate["mass"]
        A = plate["area"]
        cd = plate["cd"]
        p_pf_wrt_bf = np.asarray(plate["plate_frame_pos"], dtype=float)
        R_pf_to_bf = np.asarray(plate["plate_frame_orientation"], dtype=float)
        cop_pf = np.asarray(plate["cop_prf"], dtype=float)
        plate_lin_vel_WRF = np.asarray(plate["plate_lin_vel_WRF"], dtype=float)
        v_air_WRF = np.asarray(plate["airflow_vel_WRF"], dtype=float)

        # Weight force of the plate
        flap_weight = np.array([0,0,-m*g])

        # Rotation of the plate expressed in WRF
        R_ff_to_WRF = R_bf_to_WRF @ R_pf_to_bf

        # Normal unit vector of the plate in WRF
        n_WRF = R_ff_to_WRF[:,2]

        # Center of Pressure coordinates wrt floaty base expressed in WRF
        cop_bf = R_pf_to_bf @ cop_pf + p_pf_wrt_bf
        cop_WRF = R_bf_to_WRF @ cop_bf

        # Rigid body velocity of plate as sum of Vcom + omega_floaty_rpy x (CoP_WRF - base_WRF)
        v_flap_WRF = v_com_WRF + np.cross(rpy_rates_WRF, cop_WRF) + plate_lin_vel_WRF

        # Update force and torque vectors
        F_flap = compute_drag_force (rho, A, cd , v_flap_WRF, v_air_WRF, n_WRF) #+ flap_weight
        T_flap = np.cross(cop_WRF,F_flap)

        F += F_flap
        T += T_flap

        results.append({
            "plate": plate_name,
            "F_x": F_flap[0],
            "F_y": F_flap[1],
            "F_z": F_flap[2],
            "T_x": T_flap[0],
            "T_y": T_flap[1],
            "T_z": T_flap[2]
        })

    return F,T, results


F, T, results = drag_forces_torques(plates, [0,0,0], [0,0,0], [0,0,0], rho=AIR_DENSITY, g = 9.81)

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
