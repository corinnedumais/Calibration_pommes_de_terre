import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from tqdm import tqdm


def too_much_overlap(center, centers_list):
    x1, y1 = center
    for c in centers_list:
        x2, y2 = c
        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if d < 100:
            return True
    return False


browns = [(225 / 255, 193 / 255, 110 / 255), (218 / 255, 160 / 255, 109 / 255), (222 / 255, 184 / 255, 135 / 255),
          (225 / 255, 193 / 255, 110 / 255), (193 / 255, 154 / 255, 107 / 255)]

for it in tqdm(range(38, 41)):
    main_color = np.array(np.random.choice(range(256), size=3))
    img = np.zeros((1536, 2048, 3), dtype=np.uint8)

    for i in range(1536):
        for j in range(2048):
            if np.random.rand() > 0.12:
                img[i, j, :] = main_color
            else:
                random_color = np.array(np.random.choice(range(256), size=3))
                img[i, j, :] = random_color

    num_pdt = np.random.randint(30, 60)
    num_cible = 6

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    fig2, ax2 = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax2.imshow(np.zeros((1536, 2048)), cmap='gray')

    center_coords = []

    for i in range(num_cible):
        size = np.random.randint(30, 80)
        angle = np.random.random() * 90
        xo, yo = np.random.choice(range(300, 1700)), np.random.choice(range(250, 1250))

        while too_much_overlap((xo, yo), center_coords):
            xo, yo = np.random.choice(range(300, 1700)), np.random.choice(range(250, 1250))

        r = Rectangle((xo, yo), size, size, angle=angle)
        ax.add_artist(r)
        r.set_clip_box(ax.bbox)
        r.set_facecolor((1, 1, 1))

        r2 = Rectangle((xo, yo), size, size, angle=angle)
        ax2.add_artist(r2)
        r2.set_clip_box(ax2.bbox)
        r2.set_facecolor((1, 1, 1))

        r = Rectangle((xo, yo), size / 2, size / 2, angle=angle)
        ax.add_artist(r)
        r.set_clip_box(ax.bbox)
        r.set_facecolor((0, 0, 0))
        new_center = (xo + 1 / np.sqrt(2) * size * np.cos(np.deg2rad(45 + angle)),
                      yo + 1 / np.sqrt(2) * size * np.sin(np.deg2rad(45 + angle)))
        center_coords.append(new_center)
        r = Rectangle(new_center, size / 2, size / 2, angle=angle)
        ax.add_artist(r)
        r.set_clip_box(ax.bbox)
        r.set_facecolor((0, 0, 0))

    ax2.invert_yaxis()
    fig2.gca().set_axis_off()
    fig2.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig2.savefig(f'Synthetic_masks/syn_mask{it:03}.jpg', dpi=500)

    ells = []
    for i in range(num_pdt):
        xy = (np.random.choice(range(500, 1600)), np.random.choice(range(400, 1000)))
        while too_much_overlap(xy, center_coords):
            xy = (np.random.choice(range(500, 1600)), np.random.choice(range(400, 1000)))
        e = Ellipse(xy=xy, width=np.random.choice(range(40, 100)), height=np.random.choice(range(50, 180)),
                    angle=np.random.rand() * 360)
        center_coords.append(xy)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor(browns[np.random.randint(0, len(browns))])

    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 1536)

    ax.imshow(img)
    fig.gca().set_axis_off()
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig(f'Synthetic_data/syn{it:03}.jpg', dpi=500)
