import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math

SUBSTEPS = 4  # Number of substeps for simulation
MAX_GAP_RANGE = 0.30    # Maximum gap range for the arm

class ArmEnv(object):
    action_bound = [-1.5, 1.5]  # Action bounds for the arm's movement
    state_dim = 90  # Dimension of the state space
    action_dim = 7  # Dimension of the action space
    distance_old = 0  # Previous distance to the goal
    distance_new = 0  # Current distance to the goal
    orient_old = 0  # Previous orientation error
    orient_new = 0  # Current orientation error

    def __init__(self, sim_port=23000):
        self.offset = -0.02  # Offset for sensor placement
        self.sim_port = sim_port  # Port for connecting to the simulator
        self.connect_to_Coppeliasim()  # Establish connection to CoppeliaSim
        self.retrieve_Object_Handles()  # Retrieve object handles from the simulator
        # Target pose and initial robot arm state
        self.goal_pose = {
            'x': 1.00, 'y': 0.10, 'z': 0.70, 'pe': 0.02,  # Position and position error
            'alpha': 0, 'beta': 0, 'gamma': 0, 'oe': 60 * np.pi / 180  # Orientation and orientation error
        }
        # List of 4 predefined goal positions
        self.predefined_goals = [
            {'x': 1.00, 'y': 0.10, 'z': 0.70},
            {'x': 1.70, 'y': 0.78, 'z': 0.85},
            {'x': 2.20, 'y': 0.10, 'z': 0.90},
            {'x': 1.55, 'y': -0.55, 'z': 0.55},
        ]
        self.episode_count = 0  # Counter for episodes
        # Initially randomly select a predefined goal
        idx = np.random.randint(len(self.predefined_goals))
        self.goal_pose.update(self.predefined_goals[idx])
        self.arm_info = [0, 0, 0, -90 * np.pi / 180, 0, 90 * np.pi / 180, 0] # Initial arm joint angles
        #self.fuzzy_reward_system = FuzzyReward()  # Placeholder for fuzzy reward system
        self.on_goal = 0  # Counter for steps on goal
        self.dist_norm = 3  # Normalization factor for distance
        self.dist_norm2 = 0.05  # Secondary normalization factor
        self.safe_distance = 0.05  # Safety distance threshold
        self.sensor_range = 0.30  # Maximum sensor detection range
        self.goal_gain = 5.0  # Attractive field strength, adjustable, higher value means more reward when approaching target

    def connect_to_Coppeliasim(self):
        print(f'Program started on port {self.sim_port}')
        self.client = RemoteAPIClient(port=self.sim_port)  # Create a remote API client
        self.sim = self.client.require('sim')  # Require the simulation module
        self.client.setStepping(True)  # Enable stepping mode for the simulation
        # GUI specific: only set infinite acceleration when window exists
        try:
            self.sim.setInt32Param(self.sim.intparam_speedmodifier, 0)  # 0 = infinite acceleration
        except Exception as e:
            # 356 = headless instance, no GUI, safe to ignore
            if '356' not in str(e):
                raise
        print(f'Connected to remote API server on port {self.sim_port}')

    def retrieve_Object_Handles(self):
        self.arm_joint = {}  # Dictionary to store arm joint handles
        self.arm_joint[0] = self.sim.getObject('/LBR_iiwa_7_R800_joint1')
        self.arm_joint[1] = self.sim.getObject('/LBR_iiwa_7_R800_joint2')
        self.arm_joint[2] = self.sim.getObject('/LBR_iiwa_7_R800_joint3')
        self.arm_joint[3] = self.sim.getObject('/LBR_iiwa_7_R800_joint4')
        self.arm_joint[4] = self.sim.getObject('/LBR_iiwa_7_R800_joint5')
        self.arm_joint[5] = self.sim.getObject('/LBR_iiwa_7_R800_joint6')
        self.arm_joint[6] = self.sim.getObject('/LBR_iiwa_7_R800_joint7')
        self.goal = self.sim.getObject('/goal')  # Handle for the goal object
        self.tip = self.sim.getObject('/tip')  # Handle for the end effector
        self.target = self.sim.getObject('/target')  # Handle for the target object
        self.link_names = [
            '/LBR_iiwa_7_R800_link3', '/LBR_iiwa_7_R800_link4', '/LBR_iiwa_7_R800_link5',
            '/LBR_iiwa_7_R800_link6', '/LBR_iiwa_7_R800_link7', '/gripper_base_visible'
        ]
        self.links = [self.sim.getObject(n) for n in self.link_names]  # Retrieve link handles
        # Initialize previous frame link positions (for calculating motion direction)
        self.prev_positions = {link: self.sim.getObjectPosition(link, -1) for link in self.links}
        # Create surface sensors for each link
        self.create_surface_sensors()

    # =========================================================
    # Cone Sensor Batch Generation & Reading
    # =========================================================
    def create_surface_sensors(self, range_far=0.30, fast=True, explicit=True):
        """
        For CoppeliaSim 4.9's 5-parameter sim.createProximitySensor.
        Insert 1 pyramid volume sensor at the center of each bbox face for link3-link7 and gripper_base_visible (36 sensors total):
          • offset = 0   • range = range_far
          • x/y near = bbox face length/width
          • x/y far  = x/y near × 1.1  (slightly enlarged, 6 faces approximate 360°)
        """
        PYR_TYPE = self.sim.proximitysensor_pyramid  # constant: volume_pyramid
        SUB_TYPE = 16  # deprecated, must be 16
        OPT = (1 if explicit else 0) | (32 if fast else 0)  # bit0 + bit5

        # intParams: [faceCnt, faceCntFar, subDiv, subDivFar, rnd1, rnd2, 0, 0]
        INT_P = [4, 4, 0, 0, 0, 0, 0, 0]

        self.link_sensors = []  # List to store sensors for each link
        TYPE = self.sim.object_proximitysensor_type  # Type for proximity sensors
        for link in self.links:
            existing = self.sim.getObjectsInTree(link, TYPE)
            if existing:
                # Skip this link if sensors already exist
                print(f'[INFO] link "{self.sim.getObjectAlias(link)}" already has {len(existing)} sensors; skip.')
                self.link_sensors.append(existing)
                continue
            xmin = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_min_x)
            xmax = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_max_x)
            ymin = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_min_y)
            ymax = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_max_y)
            zmin = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_min_z)
            zmax = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_max_z)
            xm, ym, zm = (xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2  # Center of the bounding box
            xLen, yLen, zLen = xmax - xmin, ymax - ymin, zmax - zmin  # Dimensions of the bounding box

            # Six faces: (axis, sign, local Euler angles in deg)
            faces = [
                ('x', -1, (0, -90, 0)),
                ('x',  1, (0,  90, 0)),
                ('y', -1, (90, 0, 0)),
                ('y',  1, (-90, 0, 0)),
                ('z',  1, (0, 0, 0)),
                ('z', -1, (180, 0, 0))
            ]
            sensors_this_link = []  # List to store sensors for the current link
            for axis, sgn, rotDeg in faces:
                # ——Float parameter array (15)—————————————————————
                if axis == 'x':
                    xNear, yNear = zLen, yLen
                elif axis == 'y':
                    xNear, yNear = xLen, zLen
                else:  # 'z'
                    xNear, yNear = xLen, yLen
                xFar, yFar = xNear * 3, yNear * 3  # Slightly enlarge detection range
                FLT_P = [self.offset, range_far,  # offset, range
                         xNear, yNear, xFar, yFar,  # Face size parameters
                         0, 0, 0, 0, 0, 0, 0, 0, 0]  # Remaining reserved parameters set to 0

                # ——Create sensor—————————————————————————
                h = self.sim.createProximitySensor(PYR_TYPE, SUB_TYPE, OPT, INT_P, FLT_P)
                self.sim.setObjectParent(h, link, True)  # Set the sensor's parent to the link
                # Place sensor at face center (sensor vertex at obj origin when offset=0)
                pos = [xm, ym, zm]
                if axis == 'x':
                    pos[0] = xmax if sgn > 0 else xmin
                elif axis == 'y':
                    pos[1] = ymax if sgn > 0 else ymin
                else:  # 'z'
                    pos[2] = zmax if sgn > 0 else zmin
                self.sim.setObjectPosition(h, link, pos)  # Set the sensor's position
                # Set sensor orientation
                r = [math.radians(r_) for r_ in rotDeg]
                self.sim.setObjectOrientation(h, link, r)  # Set the sensor's orientation
                # Set alias for debugging
                alias = f'{self.sim.getObjectAlias(link)}_{axis}{sgn}_sensor'
                self.sim.setObjectAlias(h, alias)  # Set an alias for the sensor
                sensors_this_link.append(h)
            self.link_sensors.append(sensors_this_link)
        print(f'[4.9] created {sum(len(s) for s in self.link_sensors)} pyramid sensors')

    def get_link_directions(self):
        """
        Get the motion direction (unit vector list) of each link at the current moment.
        Returns a list corresponding to self.links, where each element is the unit vector [vx, vy, vz] of that link's velocity direction.
        Returns [0.0, 0.0, 0.0] if the link's linear velocity is zero.
        """
        directions = []  # List to store direction vectors for each link
        for link in self.links:
            new_pos = self.sim.getObjectPosition(link, -1)  # Get current position of the link
            old_pos = self.prev_positions[link]  # Get previous position of the link
            delta = [new_pos[i] - old_pos[i] for i in range(3)]  # Calculate position change
            norm = math.sqrt(sum(d * d for d in delta))  # Calculate the norm of the delta vector
            if norm > 1e-6:
                dir_vec = [d / norm for d in delta]  # Normalize the delta vector to get direction
            else:
                dir_vec = [0.0, 0.0, 0.0]  # If no movement, direction is zero
            # Update prev_positions to current frame position
            self.prev_positions[link] = new_pos
            directions.append(dir_vec)
        return directions

    def step(self, action):
        """
        Execute one action and get state, calculate reward through new interface.
        """
        done = False  # Flag to indicate if the episode is done
        action = action * np.pi / 180  # Convert input angles to radian increments
        self.arm_info = self.arm_info + action  # Update arm joint angles

        # Limit each joint angle within allowed range
        self.arm_info[0] = np.clip(self.arm_info[0], -170 * np.pi / 180, 170 * np.pi / 180)
        self.arm_info[1] = np.clip(self.arm_info[1], -120 * np.pi / 180, 120 * np.pi / 180)
        self.arm_info[2] = np.clip(self.arm_info[2], -170 * np.pi / 180, 170 * np.pi / 180)
        self.arm_info[3] = np.clip(self.arm_info[3], -120 * np.pi / 180, 120 * np.pi / 180)
        self.arm_info[4] = np.clip(self.arm_info[4], -170 * np.pi / 180, 170 * np.pi / 180)
        self.arm_info[5] = np.clip(self.arm_info[5], -120 * np.pi / 180, 120 * np.pi / 180)
        self.arm_info[6] = np.clip(self.arm_info[6], -175 * np.pi / 180, 175 * np.pi / 180)

        # Set new angles for each joint and advance simulation
        for i in range(7):
            self.sim.setJointPosition(self.arm_joint[i], self.arm_info[i])
        for _ in range(SUBSTEPS):
            self.sim.step()  # Advance simulation by SUBSTEPS×dt time

        # Get TCP (end effector) pose
        finger_xyz = self.sim.getObjectPosition(self.tip, -1)
        finger_orient = self.sim.getObjectOrientation(self.tip, self.goal)

        # Calculate normalized position and orientation errors
        dist1 = [
            (self.goal_pose['x'] - finger_xyz[0]) / self.dist_norm,
            (self.goal_pose['y'] - finger_xyz[1]) / self.dist_norm,
            (self.goal_pose['z'] - finger_xyz[2]) / self.dist_norm
        ]
        dist2 = [
            finger_orient[0] / np.pi,
            finger_orient[1] / np.pi,
            finger_orient[2] / np.pi
        ]
        distance_tipgoal = math.sqrt((self.goal_pose['x'] - finger_xyz[0])**2 +
                                     (self.goal_pose['y'] - finger_xyz[1])**2 +
                                     (self.goal_pose['z'] - finger_xyz[2])**2)
        delta_pos = math.sqrt(dist1[0]**2 + dist1[1]**2 + dist1[2]**2)  # Position error
        delta_orient = (abs(dist2[0]) + abs(dist2[1]) + abs(dist2[2]))/3  # Orientation error
        link_dirs = self.get_link_directions()  # Get current motion direction of each link
        global_min_dist = self.sensor_range  # Initialize global minimum distance

        # Calculate tip->goal unit vector
        goal_vec = [
            self.goal_pose['x'] - finger_xyz[0],
            self.goal_pose['y'] - finger_xyz[1],
            self.goal_pose['z'] - finger_xyz[2]
        ]
        g_norm = math.sqrt(sum(v * v for v in goal_vec))  # Norm of the goal vector
        goal_dir = [v / g_norm for v in goal_vec]  # Normalize the goal vector
        gripper_dir = link_dirs[-2]  # Direction of the gripper
        dp_goal = (
                goal_dir[0] * gripper_dir[0] +
                goal_dir[1] * gripper_dir[1] +
                goal_dir[2] * gripper_dir[2]
        )

        # Calculate reward
        r = -np.log2(distance_tipgoal) # Base reward based on position
        self.distance_old = self.distance_new
        if distance_tipgoal < 0.1:
            r += 1 / (distance_tipgoal ** 0.5) * dp_goal/10  # Reward for being close to the goal
            r += -np.log2(delta_orient)  # Reward for reducing orientation error
            self.orient_new = abs(dist2[0]) + abs(dist2[1]) + abs(dist2[2])  # Update orientation error
            if self.orient_new < self.orient_old:
                r += 0.5  # Reward for reducing orientation error
            elif self.orient_new > self.orient_old:
                r -= 0.5  # Penalty for increasing orientation error
            self.orient_old = self.orient_new
        # Build nearest obstacle information for each link (7 dimensions) and integrate per-sensor obstacle avoidance reward in this loop
        nearest_info = []  # List to store nearest obstacle information
        num_links = len(self.link_sensors)
        for link_idx, sensors in enumerate(self.link_sensors):
            # the last two links  have 3 sensors data, others have 1 sensor data
            k = 1 if link_idx < num_links - 2 else 3
            #  (dist, px, py, pz, sensor_handle)
            topk = []
            # Traverse all sensors of this link to find nearest obstacle
            for s in sensors:
                out = self.sim.handleProximitySensor(s)
                detected, dist = out[0], out[1]
               
                if detected and dist < self.safe_distance:
                    if dist < 0.01:
                        r -= 10.0
                        if dist<global_min_dist:
                            global_min_dist = dist
                    else:
                        m2 = self.sim.getObjectMatrix(s, -1)
                        px, py, pz = out[2]
                        vx = m2[0]*px + m2[1]*py + m2[2]*pz
                        vy = m2[4]*px + m2[5]*py + m2[6]*pz
                        vz = m2[8]*px + m2[9]*py + m2[10]*pz
                        norm = math.sqrt(vx*vx + vy*vy + vz*vz)
                        if norm > 1e-6:
                            world_dir = [vx/norm, vy/norm, vz/norm]
                        else:
                            world_dir = [0.0, 0.0, 0.0]
                        dp = (world_dir[0]*link_dirs[link_idx][0] +
                              world_dir[1]*link_dirs[link_idx][1] +
                              world_dir[2]*link_dirs[link_idx][2])
                        if dp > 0:
                            r += -1.0 / dist * dp / 20.0
                        elif dp < 0:
                            r += 1.0 / dist * abs(dp) / 20.0
                
                if detected:
                    px, py, pz = out[2]
                else:
                    dist, px, py, pz = self.sensor_range, 0.0, 0.0, 0.0
               
                if len(topk) < k:
                    topk.append((dist, px, py, pz, s))
                else:
                    
                    max_i = max(range(k), key=lambda i: topk[i][0])
                    if dist < topk[max_i][0]:
                        topk[max_i] = (dist, px, py, pz, s)
            # If less than k sensors detected, fill with max distance
            while len(topk) < k:
                topk.append((self.sensor_range, 0.0, 0.0, 0.0, None))
            # Sort by distance to get the nearest k sensors
            topk.sort(key=lambda x: x[0])
            for dist, px, py, pz, s in topk:
                # Calculate distance to the nearest obstacle
                if s is not None:
                    m = self.sim.getObjectMatrix(s, -1)
                    world_point = [
                        m[3] + m[0]*px + m[1]*py + m[2]*pz,
                        m[7] + m[4]*px + m[5]*py + m[6]*pz,
                        m[11] + m[8]*px + m[9]*py + m[10]*pz
                    ]
                    link_pos = self.sim.getObjectPosition(self.links[link_idx], -1)
                    vector = [world_point[i] - link_pos[i] for i in range(3)]
                else:
                    vector = [0.0, 0.0, 0.0]
                    dist = self.sensor_range
                
                link_dir = link_dirs[link_idx]
              
                nearest_info.extend([
                    vector[0], vector[1], vector[2],
                    link_dir[0], link_dir[1], link_dir[2],
                    dist / self.dist_norm
                ])
        sensor_data = nearest_info

        # Goal detection: task is considered complete if TCP stays in target area for certain steps
        if (self.goal_pose['x'] - self.goal_pose['pe'] < finger_xyz[0] < self.goal_pose['x'] + self.goal_pose['pe'] and
            self.goal_pose['y'] - self.goal_pose['pe'] < finger_xyz[1] < self.goal_pose['y'] + self.goal_pose['pe'] and
            self.goal_pose['z'] - self.goal_pose['pe'] < finger_xyz[2] < self.goal_pose['z'] + self.goal_pose['pe'] and
            -self.goal_pose['oe'] < finger_orient[0] < self.goal_pose['oe'] and
            -self.goal_pose['oe'] < finger_orient[1] < self.goal_pose['oe'] and
            -self.goal_pose['oe'] < finger_orient[2] < self.goal_pose['oe']):
            self.on_goal += 1
            r += 10  # Give reward for each timestep in target area
            if self.on_goal >= 50:
                done = True  # Mark episode as done if on goal for sufficient steps
        else:
            self.on_goal = 0

        # Build state vector (arm state + goal/end pose difference + on_goal flag + sensor data)
        s = np.concatenate((
            self.arm_info,
            np.array(finger_xyz) / self.dist_norm,
            np.array([self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]) / self.dist_norm,
            dist1, dist2,
            [1. if self.on_goal else 0.],
            np.array(sensor_data)
        ))
        # Return state, reward, termination flag, error info and global minimum distance for this step
        return s, r, done, delta_pos, delta_orient, global_min_dist

    def test_with_trained_trajectory(self, joint_angle):
        """
        Test model using trained trajectory.
        """
        # Set specific goal position
        self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z'] = 0.97, 0.1, 0.71
        goal_pos = np.array([self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]).tolist()
        self.sim.setObjectPosition(self.goal, -1, goal_pos)
        # Set robot arm joints to provided joint angles (trajectory point)
        for i in range(7):
            self.sim.setJointPosition(self.arm_joint[i], joint_angle[i])
        # Advance simulation to update physics and sensor states
        for _ in range(SUBSTEPS):
            self.sim.step()
        # Get current end effector pose and error relative to target
        finger_xyz = self.sim.getObjectPosition(self.tip, -1)
        finger_orient = self.sim.getObjectOrientation(self.tip, self.goal)
        dist1 = [
            (self.goal_pose['x'] - finger_xyz[0]) / self.dist_norm,
            (self.goal_pose['y'] - finger_xyz[1]) / self.dist_norm,
            (self.goal_pose['z'] - finger_xyz[2]) / self.dist_norm
        ]
        dist2 = [
            finger_orient[0] / np.pi,
            finger_orient[1] / np.pi,
            finger_orient[2] / np.pi
        ]
        delta_pos = math.sqrt(dist1[0]**2 + dist1[1]**2 + dist1[2]**2)  # Position error
        delta_orient = (abs(dist2[0]) + abs(dist2[1]) + abs(dist2[2])) / 3  # Orientation error
        # Calculate minimum distance to obstacles (for collision evaluation)
        min_dist = self.sensor_range
        for sensors in self.link_sensors:
            for s in sensors:
                out = self.sim.readProximitySensor(s)
                if out[0] and out[1] < min_dist:
                    min_dist = out[1]
        return delta_pos, delta_orient, min_dist

    def reset(self):
        """
        Reset environment and initialize object positions through new interface.
        """
        # Episode count and randomly switch predefined goal every 1 episodes
        self.episode_count += 1
        if self.episode_count % 1 == 0:
            idx = np.random.choice(len(self.predefined_goals))
            self.goal_pose.update(self.predefined_goals[idx])
        goal_pos = np.array([self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]).tolist()
        self.on_goal = 0  # Reset on_goal counter
        self.arm_info = [90 * np.pi / 180, 0, 0, -90 * np.pi / 180, 0, 90 * np.pi / 180, 0] # Reset arm joint angles
        # Set robot arm joints and target object positions
        for k in range(7):
            self.sim.setJointPosition(self.arm_joint[k], self.arm_info[k])
        self.sim.setObjectPosition(self.goal, -1, goal_pos)
        # Advance simulation one step to update sensors
        for _ in range(SUBSTEPS):
            self.sim.step()
        # Reset prev_positions for each link (since direct position setting means no motion)
        self.prev_positions = {link: self.sim.getObjectPosition(link, -1) for link in self.links}
        # Get end effector position and orientation
        finger_xyz = self.sim.getObjectPosition(self.tip, -1)
        finger_orient = self.sim.getObjectOrientation(self.tip, self.goal)
        # Calculate normalized position and orientation errors
        dist1 = [
            (self.goal_pose['x'] - finger_xyz[0]) / self.dist_norm,
            (self.goal_pose['y'] - finger_xyz[1]) / self.dist_norm,
            (self.goal_pose['z'] - finger_xyz[2]) / self.dist_norm
        ]
        dist2 = [
            finger_orient[0] / np.pi,
            finger_orient[1] / np.pi,
            finger_orient[2] / np.pi
        ]
        # Get nearest obstacle information (7 dimensions) for each link
        link_dirs = self.get_link_directions()
        nearest_info = []
        num_links = len(self.link_sensors)
        for link_idx, sensors in enumerate(self.link_sensors):
            # the last two links  have 3 sensors data, others have 1 sensor data
            k = 1 if link_idx < num_links - 2 else 3
            #  (dist, px, py, pz, sensor_handle)
            topk = []
            for s in sensors:
                out = self.sim.readProximitySensor(s)
                detected, dist = out[0], out[1]
                if detected:
                    px, py, pz = out[2]
                else:
                    dist, px, py, pz = self.sensor_range, 0.0, 0.0, 0.0
                if len(topk) < k:
                    topk.append((dist, px, py, pz, s))
                else:
                    max_i = max(range(k), key=lambda i: topk[i][0])
                    if dist < topk[max_i][0]:
                        topk[max_i] = (dist, px, py, pz, s)
            while len(topk) < k:
                topk.append((self.sensor_range, 0.0, 0.0, 0.0, None))
            topk.sort(key=lambda x: x[0])
            for dist, px, py, pz, s in topk:
                if s is not None:
                    m = self.sim.getObjectMatrix(s, -1)
                    world_point = [
                        m[3] + m[0]*px + m[1]*py + m[2]*pz,
                        m[7] + m[4]*px + m[5]*py + m[6]*pz,
                        m[11] + m[8]*px + m[9]*py + m[10]*pz
                    ]
                    link_pos = self.sim.getObjectPosition(self.links[link_idx], -1)
                    vector = [world_point[i] - link_pos[i] for i in range(3)]
                else:
                    vector = [0.0, 0.0, 0.0]
                    dist = self.sensor_range
                link_dir = link_dirs[link_idx]
                nearest_info.extend([
                    vector[0], vector[1], vector[2],
                    link_dir[0], link_dir[1], link_dir[2],
                    dist / self.dist_norm
                ])
        sensor_data = nearest_info
        # Build initial state (excluding distance_Cylinder)
        s = np.concatenate((
            self.arm_info,
            np.array(finger_xyz) / self.dist_norm,
            np.array([self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]) / self.dist_norm,
            dist1, dist2,
            [1. if self.on_goal else 0.],
            np.array(sensor_data)
        ))
        return s