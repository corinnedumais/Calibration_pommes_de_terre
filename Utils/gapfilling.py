import numpy as np
from skimage.draw import line
from matplotlib import pyplot as plt
from scipy.ndimage import binary_hit_or_miss


def gapfilling(pred_contour):
    # Step 1: Locate end points with hit or miss and label them '2' in contour map (0 is background and 1 is skeleton)
    pred_contour = mark_end_points(pred_contour)

    for _ in range(12):
        # Get position of all end points
        end_points = [(x, y) for x, y in zip(np.where(pred_contour == 2)[0], np.where(pred_contour == 2)[1])]

        # Step 2: Search for pair of end points that are within d px of each other.
        for p1 in end_points:
            for p2 in end_points:
                if p1 != p2:
                    d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    if d < 30:
                        rr, cc = line(p1[0], p1[1], p2[0], p2[1])
                        pred_contour[rr, cc] = 3

        end_points = [(x, y) for x, y in zip(np.where(pred_contour == 2)[0], np.where(pred_contour == 2)[1])]

        for point in end_points:
            current_point = point
            prev_point = None
            for i in range(20):
                x, y = current_point
                if not (0 < x < pred_contour.shape[0] - 1 and 0 < y < pred_contour.shape[1] - 1):
                    pred_contour[x, y] = 3
                    break
                else:
                    if pred_contour[x - 1, y] != 0 and (x - 1, y) != prev_point:
                        current_point = (x - 1, y)
                    elif pred_contour[x + 1, y] != 0 and (x + 1, y) != prev_point:
                        current_point = (x + 1, y)
                    elif pred_contour[x, y - 1] != 0 and (x, y - 1) != prev_point:
                        current_point = (x, y - 1)
                    elif pred_contour[x, y + 1] != 0 and (x, y + 1) != prev_point:
                        current_point = (x, y + 1)
                    elif pred_contour[x - 1, y - 1] != 0 and (x - 1, y - 1) != prev_point:
                        current_point = (x - 1, y - 1)
                    elif pred_contour[x + 1, y + 1] != 0 and (x + 1, y + 1) != prev_point:
                        current_point = (x + 1, y + 1)
                    elif pred_contour[x - 1, y + 1] != 0 and (x - 1, y + 1) != prev_point:
                        current_point = (x - 1, y + 1)
                    elif pred_contour[x + 1, y - 1] != 0 and (x + 1, y - 1) != prev_point:
                        current_point = (x + 1, y - 1)
                    prev_point = (x, y)

            k = 0.8
            xf, yf = int((point[0] - current_point[0]) / k + current_point[0]), int((point[1] - current_point[1]) / k + current_point[1])
            if 0 < xf < pred_contour.shape[0] - 1 and 0 < yf < pred_contour.shape[1] - 1:
                rr, cc = line(point[0], point[1], xf, yf)
                pred_contour[rr, cc] = 3
                pred_contour[xf, yf] = 2
            else:
                pred_contour[point[0], point[1]] = 3

        # for point in end_points:
        #     x, y = point
        #     # Old end point gets assigned 3
        #     pred_contour[x, y] = 3
        #     if 0 < x < pred_contour.shape[0] - 1 and 0 < y < pred_contour.shape[1] - 1:
        #         if pred_contour[x - 1, y] == 3:
        #             pred_contour[x + 1, y] = 2 if pred_contour[x + 1, y] == 0 else 3
        #         elif pred_contour[x + 1, y] == 3:
        #             pred_contour[x - 1, y] = 2 if pred_contour[x - 1, y] == 0 else 3
        #         elif pred_contour[x, y + 1] == 3:
        #             pred_contour[x, y - 1] = 2 if pred_contour[x, y - 1] == 0 else 3
        #         elif pred_contour[x, y - 1] == 3:
        #             pred_contour[x, y + 1] = 2 if pred_contour[x, y + 1] == 0 else 3
        #         elif pred_contour[x - 1, y - 1] == 3:
        #             pred_contour[x + 1, y + 1] = 2 if pred_contour[x + 1, y + 1] == 0 else 3
        #         elif pred_contour[x + 1, y + 1] == 3:
        #             pred_contour[x - 1, y - 1] = 2 if pred_contour[x - 1, y - 1] == 0 else 3
        #         elif pred_contour[x + 1, y - 1] == 3:
        #             pred_contour[x - 1, y + 1] = 2 if pred_contour[x - 1, y + 1] == 0 else 3
        #         elif pred_contour[x - 1, y + 1] == 3:
        #             pred_contour[x + 1, y - 1] = 2 if pred_contour[x + 1, y - 1] == 0 else 3
        #         elif pred_contour[x - 1, y] == 1:
        #             pred_contour[x + 1, y] = 2 if pred_contour[x + 1, y] == 0 else 3
        #         elif pred_contour[x + 1, y] == 1:
        #             pred_contour[x - 1, y] = 2 if pred_contour[x - 1, y] == 0 else 3
        #         elif pred_contour[x, y + 1] == 1:
        #             pred_contour[x, y - 1] = 2 if pred_contour[x, y - 1] == 0 else 3
        #         elif pred_contour[x, y - 1] == 1:
        #             pred_contour[x, y + 1] = 2 if pred_contour[x, y + 1] == 0 else 3
        #         elif pred_contour[x - 1, y - 1] == 1:
        #             pred_contour[x + 1, y + 1] = 2 if pred_contour[x + 1, y + 1] == 0 else 3
        #         elif pred_contour[x + 1, y + 1] == 1:
        #             pred_contour[x - 1, y - 1] = 2 if pred_contour[x - 1, y - 1] == 0 else 3
        #         elif pred_contour[x + 1, y - 1] == 1:
        #             pred_contour[x - 1, y + 1] = 2 if pred_contour[x - 1, y + 1] == 0 else 3
        #         elif pred_contour[x - 1, y + 1] == 1:
        #             pred_contour[x + 1, y - 1] = 2 if pred_contour[x + 1, y - 1] == 0 else 3

        # plt.subplots(figsize=(8, 8))
        # plt.imshow(pred_contour, cmap='gist_ncar')
        # plt.tight_layout()
        # plt.show()

    return pred_contour


def mark_end_points(pred_contour):
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
        end_points = binary_hit_or_miss(pred_contour, structure1=u).astype(np.int)
        pred_contour[end_points == 1] = 2

    return pred_contour

