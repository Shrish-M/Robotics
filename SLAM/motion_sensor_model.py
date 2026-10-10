import matplotlib.pyplot as plt

from helper_functions import * 

#robot node class
class robot_node:
    def __init__(self, x, y, theta):
        self.x_ = x
        self.y_ = y
        self.theta_ = theta
        self.lidar_ = None
        self.landmark_ = None
        self.pairs_ = None
        self.wall_pairs_ = None

    def add_lidar_scan(self, lidar):
        self.lidar_ = lidar

    def add_detected_landmark(self, landmark):
        self.landmark_ = landmark

    def add_landmark_pairs(self, pairs):
        self.pairs_ = pairs

def location_reader(filename = None):
    with open(filename) as f:
        lines = f.readlines()
        return [(float(line[:-1].split()[0]), float(line[:-1].split()[1])) for line in lines]

def robot_motion_reader(filename = None):
    with open(filename) as f:
        lines = f.readlines()
        return [(float(line[:-1].split()[0]), float(line[:-1].split()[1])) for line in lines]

def motion_model_calculation(robot_motion):
    robot_node_list = []
    theta = 0
    x = 1
    y = 0
    for v, va in robot_motion:
        if va != 0 and v != 0:
            alpha = (va * 0.5) % (2*math.pi)
            rot_radius = (v * 0.5) / alpha
            sin_alpha_by_2 = math.sin(alpha / 2)
            d = rot_radius * sin_alpha_by_2

            delta_x = 2 * d * math.cos(alpha / 2)
            delta_y = 2 * d * sin_alpha_by_2

            cost0 = math.cos(theta)
            sint0 = math.sin(theta)

            act_delta_x = delta_x * cost0 - delta_y * sint0
            act_delta_y = delta_y * cost0 + delta_x * sint0

            # solving out matrix multiplication to get it into
            x += act_delta_x
            y += act_delta_y
            theta += alpha
        elif v != 0 and va == 0:
            delta_x = v * 0.5
            delta_y = 0
            cost0 = math.cos(theta)
            sint0 = math.sin(theta)
            act_delta_x = delta_x * cost0 - delta_y * sint0
            act_delta_y = delta_y * cost0 + delta_x * sint0

            x += act_delta_x
            y += act_delta_y

        robot_node_list.append(robot_node(x, y, theta))

    return robot_node_list


def lidar_scan_reader(robot_node_list, filename = None):
    with open(filename, "r") as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            line = line[:-1]
            readings = line.split()
            robot_node_list[i].add_lidar_scan([3.5 if reading == "10.0000" else float(reading) for reading in readings])
            i += 1

def plot_landmarks(xs, ys, x, y):
    for x1, y1 in zip(xs, ys):
        plt.plot([x, x + x1], [y, y + y1], color='green', linewidth=1)

    plt.scatter(x + xs, y + ys, color='black', s=20, label='Line endings')

    plt.scatter(x, y, color='blue', marker='x', label='Sensor')

    plt.xlabel("x")
    plt.ylabel("y")
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    plt.title("LiDAR Scan Visualization")


def landmark_detection(robot_node_list):
    for i in range(0, len(robot_node_list)):
        lidar_scan = robot_node_list[i].lidar_
        amin = lidar_scan[0]
        amax = lidar_scan[1]
        delta_a = lidar_scan[2]

        lidar_scan = lidar_scan[3:]

        xs, ys = lidar_scan_helper(lidar_scan, amin, amax, delta_a, robot_node_list[i].theta_)

        x_pos, y_pos = robot_node_list[i].x_, robot_node_list[i].y_

        # print(x_pos, y_pos)

        plot_landmarks(xs, ys, x_pos, y_pos)
        plt.legend()
        plt.show()

        x_vals = range(1, len(lidar_scan) - 1)
        y_vals = []
        notable_grads = []

        # computing gradients
        for j in x_vals:
            grad = (lidar_scan[j + 1] - lidar_scan[j - 1]) / 2.0
            y_vals.append(grad)

            if abs(grad) > 0.15:
                if not notable_grads or notable_grads[-1][0] != j - 1:
                    notable_grads.append((j, grad))

        stack = []
        landmark_indices = []

        for x, y in notable_grads:
            if not stack:
                stack.append((x, y))
            elif stack[-1][1] < 0 < y:
                landmark_indices.append((stack[-1][0], x))
                stack.pop()
            else:
                stack.append((x, y))


        if len(landmark_indices) < 6:
            # trying wrap-around logic
            while stack:
                k = 0
                new_i = len(stack) - 1
                # checking for wrap-around first
                if stack[new_i][1] < 0 and stack[k][1] > 0:
                    left = stack[new_i][0]
                    right = stack[k][0]
                    landmark_indices.append((left, right))

                    # reducing stack size on both sides
                    stack = stack[:new_i]
                    if len(stack) > 1:
                        stack = stack[1:]
                    if len(landmark_indices) == 6:
                        break

                # the following conditions want to look at the two indices which aren't covered in the gradient plot, and have same signs on either side but huge values
                elif (stack[0][1] * notable_grads[-1][1]) > 2 and notable_grads[-1][0] > 350 and stack[0][0] < 5:
                    left = 0 if stack[0][0] != 0 else notable_grads[-1][0]
                    right = stack[0][0]

                    if stack[-1][0] == notable_grads[-1][0]:
                        stack = stack[:-1]
                        if len(stack) > 1:
                            stack = stack[1:]
                    elif len(stack) > 1:
                        stack = stack[1:]

                    landmark_indices.append((left, right))
                    if len(landmark_indices) == 6:
                        break
                else:
                    break



            # print(stack)
            # print(landmark_indices)

            # plt.plot(x_vals, y_vals)
            # plt.show()


        plot_gradients(x_vals, y_vals, landmark_indices, stack)
        landmark_pos = []

        for k, j in landmark_indices:
            # print("k, j:", (k, j))

            middle_idx = 0

            if  k <= j:
                middle_idx = (k+j) / 2.0
            else:
                middle_idx = (k + (((j-k) % 360) / 2.0)) % 360

            middle_angle = (amin + middle_idx * delta_a + robot_node_list[i].theta_) % (2 * np.pi)

            # print("Angles", middle_angle, robot_node_list[i].theta_)

            avg_dist = ((lidar_scan[k] + lidar_scan[j]) / 2) + 0.15

            # print("Dist", avg_dist)

            landmark_x = x_pos + avg_dist * np.cos(middle_angle)
            landmark_y = y_pos + avg_dist * np.sin(middle_angle)

            landmark_pos.append((landmark_x, landmark_y))

            # print("Landmark pos", landmark_x, landmark_y)

        # print("Landmark indices:", landmark_indices)
        landmarks_x = [p[0] for p in landmark_pos]
        landmarks_y = [p[1] for p in landmark_pos]
        plot_landmarks(xs, ys, x_pos, y_pos)
        plt.scatter(landmarks_x, landmarks_y, color='blue', marker='*', s=150, label='Landmarks')
        plt.legend()
        plt.xlim(-3, 6)
        print("I:", i)
        plt.show()

        robot_node_list[i].add_detected_landmark(landmark_pos)


def lidar_scan_helper(lidar_scan, amax, amin, delta_a, theta):
    amin = (amin + theta) % (2 * np.pi)
    amax = (amax + theta) % (2 * np.pi)

    angles = amin + np.arange(len(lidar_scan)) * delta_a
    angles = angles % (2 * np.pi)

    xs = lidar_scan * np.cos(angles)
    ys = lidar_scan * np.sin(angles)

    return xs, ys


def plot_gradients(x_vals, y_vals, landmark_indices, stack):
    # print(stack)
    # print(landmark_indices)
    plt.plot(x_vals, y_vals)
    plt.xlabel("x (indices)")
    plt.ylabel("y (gradients)")
    plt.show()


def pair_landmarks(robot_node_list):

    for i in range(0, len(robot_node_list)):
        robot_node = robot_node_list[i]
        landmarks = robot_node.landmark_
        pairs = []
        gt_landmarks = gt_landmark.copy()
        for landmark in landmarks:
            square_distances = [(landmark[0] - x) * (landmark[0] - x) + (landmark[1] - y) * (landmark[1] - y) for x, y in gt_landmarks]
            distance_dict = {dist : i for i, dist in enumerate(square_distances)}
            square_distances.sort()
            if square_distances[0] <= 1.0:
                index = distance_dict[square_distances[0]]
                g_landmark = (gt_landmarks[index][0], gt_landmarks[index][1])
                gt_landmarks.pop(index)
                pairs.append([landmark, g_landmark])

        # landmark_x = [p[0][0] for p in pairs]
        # landmark_y = [p[0][1] for p in pairs]
        #
        # g_landmark_x = [p[1][0] for p in pairs]
        # g_landmark_y = [p[1][1] for p in pairs]
        #
        # plt.figure(figsize=(6, 6))
        #
        # plt.scatter(landmark_x, landmark_y, marker='*', color='blue', s=50, label='Landmarks')
        #
        # plt.scatter(g_landmark_x, g_landmark_y, marker='*', color='red', s=50, label='Ground Truth Landmarks')
        #
        # for lx, ly, gx, gy in zip(landmark_x, landmark_y, g_landmark_x, g_landmark_y):
        #     plt.plot([lx, gx], [ly, gy], color='gray', linestyle='-', linewidth=0.8)
        #
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.xlim(-2, 6)
        # plt.ylim(-3, 3)
        # plt.legend()
        # plt.show()

        robot_node_list[i].add_landmark_pairs(pairs)

#****************************************************************************************


if __name__ == '__main__':

    gt_location = location_reader('location.txt')
    robot_motion = robot_motion_reader('robot_motion.txt')
    robot_node_list = motion_model_calculation(robot_motion)

    # draw_robot_trajectory(robot_node_list, gt_location)

    lidar_data = lidar_scan_reader(robot_node_list, 'lidar_scan.txt')
    landmark_detection(robot_node_list)

    pair_landmarks(robot_node_list)




