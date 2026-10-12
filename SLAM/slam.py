
import numpy as np
from numpy.ma.core import true_divide

from helper_functions import * 

def estimate_transform(left_list, right_list, fix_scale = False):
    l_points = np.asarray(left_list)
    r_points = np.asarray(right_list)

    if len(l_points) < 2:
        return 1, 1.0, 0.0, 0.0, 0.0

    # computing COM for both
    l_mean = np.mean(l_points, axis = 0)
    r_mean = np.mean(r_points, axis = 0)

    # common
    l_centered = l_points - l_mean
    r_centered = r_points - r_mean


    # calculating R matrix and getting cos and sin values
    S_numerator = np.sum(r_centered[:, 0] * l_centered[:, 0] + r_centered[:,1] * l_centered[:, 1])
    S_denominator = np.sum(-1 * r_centered[:, 0] * l_centered[:, 1] + r_centered[:, 1] * l_centered[:, 0])

    norm = np.sqrt(S_numerator * S_numerator + S_denominator * S_denominator)

    if norm == 0.0:
        cos = 1.0
        sin = 0.0
    else:
        cos = S_numerator / norm
        sin = S_denominator / norm

    R = np.array([[cos, -sin],
                  [sin, cos]])



    # calculating lambda
    if not fix_scale:

        lambda_num = np.sum(np.linalg.norm(r_centered, axis=1) ** 2)

        #we use a transpose here because we want to multiply the 2x2 R by the 2xN list of points where each column is a point, then we transpose again to get it back to an Nx2 matrix where each row is a point
        lambda_denom = np.sum(np.linalg.norm((R @ l_centered.T).T, axis=1) **2)

        if lambda_denom == 0:
            scale = 1

        else:
            scale = np.sqrt(lambda_num / lambda_denom)

    else:
        scale = 1
    # calculating tx, ty

    if norm != 0.0:
        t = r_mean - scale * (R @ l_mean)
    else:
        t = [0.0, 0.0]

    return scale, cos, sin, t[0], t[1]

def transform_estimate_and_correction(robot_node_list, robot_motion):
    first = True
    num_pairs = 0

    for i in range(0, len(robot_node_list)):
        if not first:
            # recalculate motion model
            prev_x = robot_node_list[i-1].x_
            prev_y = robot_node_list[i-1].y_
            prev_theta = robot_node_list[i-1].theta_

            new_x, new_y, new_theta = single_motion_model(prev_x, prev_y, prev_theta, robot_motion[i][0], robot_motion[i][1])
            robot_node_list[i].x_ = new_x
            robot_node_list[i].y_ = new_y
            robot_node_list[i].theta_ = new_theta

            # do landmark detection again
            single_landmark_detection(robot_node_list[i])

            # re-pair landmarks
            single_landmark_pair(robot_node_list[i])

        pairs = robot_node_list[i].pairs_

        if not pairs:
            continue

        landmarks = robot_node_list[i].landmark_
        detected_landmarks = [landmarks[pair[0]] for pair in pairs]
        corresponding_gt = [gt_landmark[pair[1]] for pair in pairs]


        new_x, new_y, new_theta = correct_pose((robot_node_list[i].x_, robot_node_list[i].y_, robot_node_list[i].theta_), estimate_transform(detected_landmarks, corresponding_gt, True))
        robot_node_list[i].x_ = new_x
        robot_node_list[i].y_ = new_y


        robot_node_list[i].theta_ = new_theta % (2 * np.pi)
        first = False

def single_motion_model(prev_x, prev_y, prev_theta, v, v_theta):
    if v_theta != 0:
        delta_theta = v_theta * 0.5
        r = v * 0.5 / abs(delta_theta)
        delta_x, delta_y = None, None
        # if v_theta >= 0:
        delta_x = 2 * r * math.sin(abs(delta_theta) / 2) * math.cos(delta_theta / 2)
        delta_y = 2 * r * math.sin(abs(delta_theta) / 2) * math.sin(delta_theta / 2)
        # else:
        #    delta_x = - 2 * r * math.sin(delta_theta / 2) * math.cos(delta_theta )
        #    delta_y = - 2 * r * math.sin(delta_theta / 2) * math.sin(delta_theta )
    else:
        delta_x = v * 0.5
        delta_y = 0

    final_theta = prev_theta + v_theta * 0.5
    final_x = prev_x + math.cos(prev_theta) * delta_x - math.sin(prev_theta) * delta_y
    final_y = prev_y + math.sin(prev_theta) * delta_x + math.cos(prev_theta) * delta_y
    return final_x, final_y, final_theta

def single_landmark_detection(robot_node):
    single_lidar_scan = robot_node.lidar_
    min_angle, max_angle, delta_angle = single_lidar_scan[:3]

    scan_rays = single_lidar_scan[3:]
    scan_rays_copy = scan_rays[:]
    scan_rays = [scan_rays[-1]] + scan_rays + [scan_rays[0]]

    gradient_data = []

    for i in range(1, len(scan_rays) - 1):
        l = scan_rays[i - 1]
        r = scan_rays[i + 1]
        derivative = (r - l) / 2.0
        gradient_data.append(derivative)

    gradient_data = gradient_data + gradient_data

    detected_landmark = []

    ray_index_sum, ray_distance_sum, ray_sum = 0, 0, 0
    inside_landmark = False

    for i in range(len(gradient_data)):
        measure = gradient_data[i]
        if measure < -0.24:
            if i >= 360: break
            ray_index_sum, ray_distance_sum, ray_sum = 0, 0, 0
            inside_landmark = True
        elif measure > 0.24:
            if inside_landmark and ray_sum:
                detected_landmark.append((ray_index_sum / ray_sum, ray_distance_sum / ray_sum))
            inside_landmark = False
            if i >= 360:
                break
        else:
            if inside_landmark:
                ray_index_sum += i
                if i < 360:
                    ray_distance_sum += scan_rays_copy[i]
                else:
                    ray_distance_sum += scan_rays_copy[i - 360]
                ray_sum += 1

    landmarks = []
    for ray_index, ray_length in detected_landmark:
        # offset
        ray_length += 0.15
        ray_angle = ray_index * delta_angle
        ray_angle += robot_node.theta_
        landmarks.append(
            (robot_node.x_ + math.cos(ray_angle) * ray_length, robot_node.y_ + math.sin(ray_angle) * ray_length))
    robot_node.add_detected_landmark(landmarks)

def single_landmark_pair(robot_node):
    landmark = robot_node.landmark_

    landmark_pairs = []

    max_radius = 1.0

    for i in range(len(landmark)):
        cx, cy = landmark[i]
        distance = []
        for rx, ry in gt_landmark:
            temp_distance = ((cx - rx) ** 2 + (cy - ry) ** 2)
            if temp_distance < max_radius ** 2:
                distance.append(temp_distance)
            else:
                distance.append(sys.maxsize)
        min_dis = min(distance)
        if min_dis != sys.maxsize:
            landmark_pairs.append((i, distance.index(min_dis)))

    robot_node.add_landmark_pairs(landmark_pairs)


def get_subsample_rays(robot_node):
    lidar = robot_node.lidar_
    return lidar[3::10]

def get_corresponding_points_on_boundary(points):
    act_points = []
    wall_points = []

    for point in points:
        # fix x, calc distance to top and bottom walls
        dist_top = 3 - point[1]
        dist_bottom = point[1] + 3

        # fix y, calc distance to left and right walls
        dist_left = point[0]
        dist_right = 6 - point[0]

        distances = [
            (dist_top, (point[0], 3)),
            (dist_bottom, (point[0], -3)),
            (dist_left, (0, point[1])),
            (dist_right, (6, point[1])),
        ]

        min_dist, wall_coord = min(distances, key=lambda x: x[0])

        if (min_dist <= 0.1):
            act_points.append(point)
            wall_points.append(wall_coord)

    return act_points, wall_points

def get_icp_transform(points, iterations):

    concat_transform = None

    for i in range(0, iterations):
        point_lists = get_corresponding_points_on_boundary(points)

        cur_transform = estimate_transform(point_lists[0], point_lists[1], True)

        if concat_transform is None:
            concat_transform = cur_transform
        else:
            concat_transform = concatenate_transform(concat_transform, cur_transform)

        points = [apply_transform(cur_transform, p) for p in points]

    return concat_transform

def lidar_scan_helper(lidar_scan, amin, delta_a, theta):
    amin = (amin + theta) % (2 * np.pi)


    angles = amin + np.arange(len(lidar_scan)) * 10 * delta_a
    angles = angles % (2 * np.pi)

    xs = lidar_scan * np.cos(angles)
    ys = lidar_scan * np.sin(angles)

    return xs, ys

def generate_points(robot_node):
    subsample = get_subsample_rays(robot_node)
    a_min = robot_node.lidar_[0]
    delta_a = robot_node.lidar_[2]

    xs, ys = lidar_scan_helper(subsample, a_min, delta_a, robot_node.theta_)

    x, y = robot_node.x_, robot_node.y_

    return [(x + xs[i], y + ys[i]) for i in range(0, len(subsample))]


def featureless_transform_estimate_and_correction(robot_node_list, robot_motion):

    for i in range(0, len(robot_node_list)):
        if i != 0:
            # recalculate motion model
            prev_x = robot_node_list[i - 1].x_
            prev_y = robot_node_list[i - 1].y_
            prev_theta = robot_node_list[i - 1].theta_

            new_x, new_y, new_theta = single_motion_model(prev_x, prev_y, prev_theta, robot_motion[i][0],
                                                          robot_motion[i][1])
            robot_node_list[i].x_ = new_x
            robot_node_list[i].y_ = new_y
            robot_node_list[i].theta_ = new_theta

        points = generate_points(robot_node_list[i])
        transform = get_icp_transform(points, 100)


        new_x, new_y, new_theta = correct_pose((robot_node_list[i].x_, robot_node_list[i].y_, robot_node_list[i].theta_), transform)
        robot_node_list[i].x_ = new_x
        robot_node_list[i].y_ = new_y
        robot_node_list[i].theta_ = new_theta

#****************************************************************************************



if __name__ == '__main__':

    gt_location = location_reader('location.txt')
    robot_motion = robot_motion_reader('robot_motion.txt')
    robot_node_list = motion_model_calculation(robot_motion)

    lidar_data = lidar_scan_reader(robot_node_list, 'lidar_scan.txt')

    robot_node_list_feature_based = copy.deepcopy(robot_node_list)

    robot_node_list_featureless = copy.deepcopy(robot_node_list)

    landmark_detection(robot_node_list_feature_based)

    pair_landmarks(robot_node_list_feature_based)
    transform_estimate_and_correction(robot_node_list_feature_based, robot_motion)

    draw_robot_trajectory(robot_node_list_feature_based, gt_location)

    featureless_transform_estimate_and_correction(robot_node_list_featureless, robot_motion)

    draw_robot_trajectory(robot_node_list_featureless, gt_location)