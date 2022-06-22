import numpy as np
from matplotlib.colors import ListedColormap
from skimage.draw import line
import time
from matplotlib import pyplot as plt
from scipy.ndimage import binary_hit_or_miss


def fill_gaps(contours, n_iterations):
    # Locate end points with hit or miss and label them '2' in contour map (0 is background and 1 is skeleton)
    contours = mark_end_points(contours)

    # plt.figure(figsize=(7, 7))
    # plt.title(f'0 itérations', fontsize=18)
    # plt.imshow(contours[400:600, 800:1000], cmap=ListedColormap(['#000000', '#ffffff', '#ff0000']))
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    for n in range(n_iterations):
        # Get position of all end points
        end_points = [(x, y) for x, y in zip(np.where(contours == 2)[0], np.where(contours == 2)[1])]

        # Pair up the points that are close enough together
        contours = join_pairs(contours, end_points, max_distance=30)

        # Get position of all new end points
        end_points = [(x, y) for x, y in zip(np.where(contours == 2)[0], np.where(contours == 2)[1])]

        # Grow each end point
        contours = grow_end_points(contours, end_points)

        # plt.figure(figsize=(7, 7))
        # plt.title(f'{n + 1} itérations', fontsize=18)
        # plt.imshow(contours[400:600, 800:1000], cmap=ListedColormap(['#000000', '#ffffff', '#ff0000', '#fff000']))
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

    return contours


def mark_end_points(contours):
    structure_S = np.zeros((3, 3))
    structure_S[1, :2] = 1
    structure_D = np.zeros((3, 3))
    structure_D[0, 0] = 1
    structure_D[1, 1] = 1
    U_structures = []
    for _ in range(4):
        U_structures.append(structure_S)
        structure_S = np.rot90(structure_S)
        U_structures.append(structure_D)
        structure_D = np.rot90(structure_D)

    for u in U_structures:
        end_points = binary_hit_or_miss(contours, structure1=u).astype(np.int)
        contours[end_points == 1] = 2

    return contours


def join_pairs(contours, end_points, max_distance):
    pairs = []
    for i, p1 in enumerate(end_points):
        for p2 in end_points[i:]:  # To avoid checking same pair twice
            if p1 != p2:
                d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if d < max_distance:
                    pairs.append([p1, p2, d])

    while pairs:
        sorted_pairs = sorted(pairs, key=lambda x: x[2])
        min_dist = sorted_pairs[0]
        rr, cc = line(min_dist[0][0], min_dist[0][1], min_dist[1][0], min_dist[1][1])
        contours[rr, cc] = 3
        for p, p2 in zip(pairs, np.array(pairs, dtype=object).T[:2].T):
            for c in p2:
                if c == min_dist[0] or c == min_dist[1]:
                    if p in pairs:
                        pairs.remove(p)

    return contours


def grow_end_points(contours, end_points):
    for point in end_points:
        current_point = point
        prev_point = None
        for i in range(15):
            x, y = current_point
            if not (0 < x < contours.shape[0] - 1 and 0 < y < contours.shape[1] - 1):
                contours[x, y] = 3
                break
            else:
                if contours[x - 1, y] != 0 and (x - 1, y) != prev_point:
                    current_point = (x - 1, y)
                elif contours[x + 1, y] != 0 and (x + 1, y) != prev_point:
                    current_point = (x + 1, y)
                elif contours[x, y - 1] != 0 and (x, y - 1) != prev_point:
                    current_point = (x, y - 1)
                elif contours[x, y + 1] != 0 and (x, y + 1) != prev_point:
                    current_point = (x, y + 1)
                elif contours[x - 1, y - 1] != 0 and (x - 1, y - 1) != prev_point:
                    current_point = (x - 1, y - 1)
                elif contours[x + 1, y + 1] != 0 and (x + 1, y + 1) != prev_point:
                    current_point = (x + 1, y + 1)
                elif contours[x - 1, y + 1] != 0 and (x - 1, y + 1) != prev_point:
                    current_point = (x - 1, y + 1)
                elif contours[x + 1, y - 1] != 0 and (x + 1, y - 1) != prev_point:
                    current_point = (x + 1, y - 1)
                prev_point = (x, y)

        k = 0.8
        xf, yf = int((point[0] - current_point[0]) / k + current_point[0]), int(
            (point[1] - current_point[1]) / k + current_point[1])
        if 0 < xf < contours.shape[0] - 1 and 0 < yf < contours.shape[1] - 1:
            rr, cc = line(point[0], point[1], xf, yf)
            contours[rr, cc] = 3
            contours[xf, yf] = 2
        else:
            contours[point[0], point[1]] = 3

    return contours
