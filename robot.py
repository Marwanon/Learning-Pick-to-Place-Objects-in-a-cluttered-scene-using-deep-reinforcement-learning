import time
import os
import numpy as np
import utils
from simulation import vrep

class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, workspace_limits,workspace_limit_place,
                 is_testing, test_preset_cases, test_preset_file, place=False):

        self.is_sim = is_sim
        self.workspace_limits = workspace_limits
        self.workspace_limit_place = workspace_limit_place
        self.place_task = place
        self.sim_home_position = [-0.3, 0.0, 0.45]
        self.sim_home_place_position = [0.07, -0.5, 0.45]

        # If in simulation...
        if self.is_sim:

            # Define colors for object meshes (Tableau palette)
            self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                           [89.0, 161.0, 79.0], # green
                                           [156, 117, 95], # brown
                                           [242, 142, 43], # orange
                                           [237.0, 201.0, 72.0], # yellow
                                           [186, 176, 172], # gray
                                           [255.0, 87.0, 89.0], # red
                                           [176, 122, 161], # purple
                                           [118, 183, 178], # cyan
                                           [255, 157, 167]])/255.0 #pink

            # Read files in object mesh directory
            self.obj_mesh_dir = obj_mesh_dir
            self.num_obj = num_obj
            self.mesh_list = os.listdir(self.obj_mesh_dir)

            # Randomly choose objects to add to scene
            self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
            self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

            # Connect to simulator
            vrep.simxFinish(-1) # Just in case, close all opened connections
            self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
            if self.sim_client == -1:
                print('Failed to connect to simulation (V-REP remote API server). Exiting.')
                exit()
            else:
                print('Connected to simulation.')
                self.restart_sim()

            self.is_testing = is_testing
            self.test_preset_cases = test_preset_cases
            self.test_preset_file = test_preset_file

            # Setup virtual camera in simulation
            self.setup_sim_camera()

            # If testing, read object meshes and poses from test case file
            if self.is_testing and self.test_preset_cases:
                file = open(self.test_preset_file, 'r')
                file_content = file.readlines()
                self.test_obj_mesh_files = []
                self.test_obj_mesh_colors = []
                self.test_obj_positions = []
                self.test_obj_orientations = []
                for object_idx in range(self.num_obj):
                    file_content_curr_object = file_content[object_idx].split()
                    self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir,file_content_curr_object[0]))
                    self.test_obj_mesh_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
                    self.test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
                    self.test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
                file.close()
                self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)

            # Add objects to simulation environment
            self.add_objects()

    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale


    def add_objects(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1], self.test_obj_positions[object_idx][2]]
                object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            if not (self.is_testing and self.test_preset_cases):
                time.sleep(0.2)
        self.prev_obj_positions = []
        self.obj_positions = []


    def restart_sim(self):

        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)


    def check_sim(self):
        buffer_meters = 0.1
        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - buffer_meters and gripper_position[0] < self.workspace_limits[0][1] + buffer_meters and gripper_position[1] > self.workspace_limits[1][0] - buffer_meters and gripper_position[1] < self.workspace_limits[1][1] + buffer_meters and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()


    def get_task_score(self):

        key_positions = np.asarray([[-0.625, 0.125, 0.0], # red
                                    [-0.625, -0.125, 0.0], # blue
                                    [-0.375, 0.125, 0.0], # green
                                    [-0.375, -0.125, 0.0]]) #yellow

        obj_positions = np.asarray(self.get_obj_positions())
        obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
        obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

        key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
        key_positions = np.tile(key_positions, (1 ,obj_positions.shape[1] ,1))

        key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
        key_nn_idx = np.argmin(key_dist, axis=0)

        return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)


    def check_goal_reached(self):

        goal_reached = self.get_task_score() == self.num_obj
        return goal_reached

    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations


    def reposition_objects(self, workspace_limits):

        # Move gripper out of the way
        self.move_to([-0.3, 0, 0.3], None)
        for object_handle in self.object_handles:
            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
            time.sleep(2)


    def get_camera_data(self):

        if self.is_sim:
            # Get color image from simulation
            sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(np.float)/255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)
            # Get depth image from simulation
            sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear
      
        return color_img, depth_img


    def close_gripper(self, asynch=False):

        if self.is_sim:
            gripper_motor_velocity = -0.5
            gripper_motor_force = 100
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            gripper_fully_closed = False
            while gripper_joint_position > -0.045: # Block until gripper is fully closed
                sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
                # print(gripper_joint_position)
                if new_gripper_joint_position >= gripper_joint_position:
                    return gripper_fully_closed
                gripper_joint_position = new_gripper_joint_position
            gripper_fully_closed = True

        return gripper_fully_closed

    def open_gripper(self, asynch=False):

        if self.is_sim:
            gripper_motor_velocity = 0.5
            gripper_motor_force = 20
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            while gripper_joint_position < 0.03: # Block until gripper is fully open
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)


    def move_to(self, tool_position, tool_orientation):

        if self.is_sim:
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)

            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_magnitude/0.02))

            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)

       
    def go_home(self):
        self.move_joints(self.home_joint_config)
    # Note: must be preceded by close_gripper()
    
    def check_grasp(self):
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        return tool_analog_input2 > 0.26

    def place_task (self,place_position, workspace_limit_place):
        workspace_limit_place = np.asarray([[-0.1, 0.3],[-0.7, -0.3],[-0.0001, 0.5]])
        if workspace_limit_place is None:
            workspace_limit_place = self.workspace_limit_place
        print('pridecting: place at (%f, %f, %f)' % (place_position[0], place_position[1], place_position[2]))
        
        place_position[2] = max(place_position[2] + 0.04 + 0.02, workspace_limit_place[2][0] + 0.02)
        drop_x = (self.workspace_limit_place[0][1] - self.workspace_limit_place[0][0] - 0.2) * np.random.random_sample() + self.workspace_limit_place[0][0] + 0.1
        drop_y = (self.workspace_limit_place[1][1] - self.workspace_limit_place[1][0] - 0.2) * np.random.random_sample() + self.workspace_limit_place[1][0] + 0.1
        place_position = [drop_x, drop_y, place_position[2]]
        place_predictions = np.asarray(place_position).copy()
        place_predictions.append(place_predictions)
        return place_predictions
    
    # Primitives ----------------------------------------------------------

    def grasp(self, position, heightmap_rotation_angle, workspace_limits, trainer, workspace_limit_place):
        
        if workspace_limits is None:
            workspace_limits = self.workspace_limits
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:
            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2
            # Avoid collision with floor
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)
            # Move gripper to location above grasp target
            grasp_location_margin = 0.15
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)
            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))
            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))
            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)
            # Ensure gripper is open
            self.open_gripper()
            # Approach grasp target
            self.move_to(position, None)
            # Close gripper to grasp target
            gripper_full_closed = self.close_gripper()
            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None)
#            self.move_to(self.sim_home_position, None)
            # Check if grasp is successful
            gripper_full_closed = self.close_gripper()
            grasp_success = not gripper_full_closed
            # Move the grasped object elsewhere
#            if grasp_success:
#                object_positions = np.asarray(self.get_obj_positions())
#                object_positions = object_positions[:, 2]
#                grasped_object_ind = np.argmax(object_positions)
#                grasped_object_handle = self.object_handles[grasped_object_ind]
#                vrep.simxSetObjectPosition(self.sim_client, grasped_object_handle, -1, (-1, -1 + 0.05 * float(grasped_object_ind), 0.1), vrep.simx_opmode_blocking)
            
            if grasp_success:
                self.place(position, heightmap_rotation_angle, trainer, workspace_limit_place)

        return grasp_success
    
    
    def place(self, place_position, heightmap_rotation_angle, trainer, workspace_limit_place=None):
        
        if workspace_limit_place is None:
            workspace_limit_place = self.workspace_limit_place
        print('Executing: place at (%f, %f, %f)' % (place_position[0], place_position[1], place_position[2]))
        
        if self.is_sim:
            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2
            # Compute tool orientation from heightmap rotation angle
            place_position[2] = max(place_position[2] + 0.04 + 0.02, workspace_limit_place[2][0] + 0.02)
            drop_x = (self.workspace_limit_place[0][1] - self.workspace_limit_place[0][0] - 0.2) * np.random.random_sample() + self.workspace_limit_place[0][0] + 0.1
            drop_y = (self.workspace_limit_place[1][1] - self.workspace_limit_place[1][0] - 0.2) * np.random.random_sample() + self.workspace_limit_place[1][0] + 0.1
            place_position = [drop_x, drop_y, place_position[2]]
            place_position = np.asarray(place_position).copy()
            # Avoid collision with floor
            place_location_margin = 0.1
            sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_place_target = (place_position[0], place_position[1], place_position[2] + place_location_margin)
            sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_place_target = (place_position[0], place_position[1], place_position[2] + place_location_margin)

            sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
            if tool_rotation_angle - gripper_orientation[1] > 0:
                 increment = 0.2
            else:
                 increment = -0.2
            while abs(tool_rotation_angle - gripper_orientation[1]) >= 0.2:
                 vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + increment, np.pi/2), vrep.simx_opmode_blocking)
                 time.sleep(0.01)
                 sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)
#            move gripper to home place position
            self.move_to(self.sim_home_place_position, None) # move gripper to home place position
            gripper_fully_closed = self.close_gripper() # return to def gripper_close to do closing (closing condition)
            place_success = not gripper_fully_closed  # place will be success if gripper is not fully close, it means there is object within gripper fingers
            place_not_success = gripper_fully_closed # place will be non-success if gripper is  fully close, it means there is no object within gripper fingers
            if place_success: #means not fully close
                self.move_to(place_position, None) # now robot will move  to the place position 
                self.open_gripper()
                self.move_to(location_above_place_target, None) # avoiding clash between gripper and floor as safety 
                self.move_to(self.sim_home_position, None) # back to the sart point to start new iteration            
                print('Place successful: ' + str(place_success))
            elif place_not_success:    # There is no object present, so we cannot possibly place! 
                self.move_to(self.sim_home_position, None)       
                print('Place successful: ' + str(place_success))
#            place_success.append(place_success)
            trainer.place_success_log.append(place_success)
            return place_success, place_not_success
           