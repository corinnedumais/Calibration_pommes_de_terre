import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from tqdm import tqdm

browns = [(225 / 255, 193 / 255, 110 / 255), (218 / 255, 160 / 255, 109 / 255), (222 / 255, 184 / 255, 135 / 255),
          (225 / 255, 193 / 255, 110 / 255), (193 / 255, 154 / 255, 107 / 255)]


def too_much_overlap(center, centers_list, max_d):
    x1, y1 = center
    for c in centers_list:
        x2, y2 = c
        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if d < max_d:
            return True
    return False


def generate_background(noise):
    main_color = np.array(np.random.choice(range(256), size=3))
    img = np.zeros((1536, 2048, 3), dtype=np.uint8)
    for i in range(1536):
        for j in range(2048):
            if np.random.rand() > noise:
                img[i, j, :] = main_color
            else:
                random_color = np.array(np.random.choice(range(256), size=3))
                img[i, j, :] = random_color
    return img


def generate_synthetic_img(no, noise=0.1, instances=70, targets=6, x_pos=range(500, 1600), y_pos=range(400, 1000),
                           widths=range(40, 100), heights=range(50, 180), max_d=100):
    # Generate background of random color with noise
    img = generate_background(noise)

    # List to keep track of center positions of objects in the image
    center_coords = []

    # Initialize plots
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})    # For images
    fig2, ax2 = plt.subplots(subplot_kw={'aspect': 'equal'})  # For masks

    # Draw targets
    for i in range(targets):
        size = np.random.randint(40, 90)
        angle = np.random.random() * 90
        xo, yo = np.random.choice(range(300, 1700)), np.random.choice(range(250, 1250))

        # Find center coordinates that dont cause too much overlap with already drawn targets
        while too_much_overlap((xo, yo), center_coords, max_d):
            xo, yo = np.random.choice(range(300, 1700)), np.random.choice(range(250, 1250))

        # Bigger white square part of the target
        r = Rectangle((xo, yo), size, size, angle=angle)
        ax.add_artist(r)
        r.set_clip_box(ax.bbox)
        r.set_facecolor((1, 1, 1))

        # Mask
        r2 = Rectangle((xo, yo), size, size, angle=angle)
        ax2.add_artist(r2)
        r2.set_clip_box(ax2.bbox)
        r2.set_facecolor((1, 1, 1))

        # Two smaller diagonal black squares part of the target
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

    # Once all the targets are drawn, save the masks
    ax2.imshow(np.zeros((1536, 2048)), cmap='gray')
    ax2.invert_yaxis()
    fig2.gca().set_axis_off()
    fig2.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig2.savefig(f'Synthetic_masks/syn_mask{no:02}.png', dpi=500)

    # Draw potatoes
    for i in range(instances):
        xy = (np.random.choice(x_pos), np.random.choice(y_pos))
        while too_much_overlap(xy, center_coords, max_d):
            xy = (np.random.choice(x_pos), np.random.choice(y_pos))
        e = Ellipse(xy=xy, width=np.random.choice(widths), height=np.random.choice(heights),
                    angle=np.random.rand() * 360)
        center_coords.append(xy)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor(browns[np.random.randint(0, len(browns))])

    # Once all the instances are drawn, save the image
    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 1536)
    ax.imshow(img)
    fig.gca().set_axis_off()
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig(f'Synthetic_data/syn{no:02}.jpg', dpi=500)


for it in tqdm(range(6, 51)):

    # Generate random parameters for each image
    noise = np.random.random() * 0.15
    num_pdt = np.random.randint(40, 75)
    targets = np.random.randint(4, 9)
    x_inf, x_sup = np.random.randint(100, 500), np.random.randint(1500, 2000)
    y_inf, y_sup = np.random.randint(100, 500), np.random.randint(1000, 1500)
    w_inf, w_sup = np.random.randint(80, 120), np.random.randint(130, 140)
    h_inf, h_sup = np.random.randint(140, 170), np.random.randint(170, 210)
    max_d = np.random.randint(50, 100)

    # Create and save the synthetic image
    generate_synthetic_img(no=it,
                           noise=noise,
                           instances=num_pdt,
                           targets=targets,
                           x_pos=range(x_inf, x_sup),
                           y_pos=range(y_inf, y_sup),
                           widths=range(w_inf, w_sup),
                           heights=range(h_inf, h_sup),
                           max_d=max_d)