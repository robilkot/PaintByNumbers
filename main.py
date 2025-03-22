import cv2
import numpy as np
from scipy.spatial import KDTree


def extract_palette(image, num_colors):
    reshaped = image.reshape(-1, 3)
    colors, counts = np.unique(reshaped, axis=0, return_counts=True)
    top_colors = colors[np.argsort(-counts)[:num_colors]]
    return [tuple(color) for color in top_colors]


def load_palette(filename="colors.txt"):
    with open(filename, "r") as file:
        return [tuple(int(line[i:i + 2], 16) for i in (4, 2, 0)) for line in file.read().splitlines() if not line[:6].__contains__('#')]


def apply_median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)


def quantize_colors(image, palette):
    reshaped = image.reshape(-1, 3)
    tree = KDTree(palette)
    _, indices = tree.query(reshaped)
    quantized = np.array(palette)[indices].reshape(image.shape)
    return quantized


def draw_color_contours(source, target, color_dict, min_area, text_color=(0, 0, 0), contours_color=(170, 170, 170)):
    output = source.copy() if target is None else target

    results = list(get_color_contours(source, color_dict, min_area))
    contours = list([res[0] for res in results])
    circles = list([res[1] for res in results])
    numbers = list([res[2] for res in results])
    results = zip(contours, circles, numbers)

    cv2.drawContours(output, contours, -1, contours_color, 1)

    font_face = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.25
    thickness = 1
    spacing = 5.5 * font_scale

    for contour, (center, radius), number in results:
        cv2.circle(output, center, int(radius), (0, 0, 255), 1)
        cv2.circle(output, center, 2, (0, 0, 255), 2)

        # draw single characters to configure spacing
        color_str = str(number)
        text = [str(c) for c in color_str]
        text_width, text_height = cv2.getTextSize(color_str, font_face, font_scale, thickness)[0]
        text_origin = (center[0] - int(text_width / 2), center[1] + int(text_height / 2))

        for k, char in enumerate(text):
            coords_x, coords_y = text_origin
            cv2.putText(output, char, (int(coords_x + k * spacing), int(coords_y)), font_face, font_scale,
                        text_color,
                        thickness, cv2.LINE_AA)

    return output


def get_color_contours(source, color_dict, min_area):
    for color, number in color_dict.items():
        mask = cv2.inRange(source, np.array(color), np.array(color))

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        for i, contour in enumerate(contours):
            # child contour
            if hierarchy[0][i][3] != -1:
                continue

            # Skip small contours based on area
            if cv2.contourArea(contour) < min_area:
                continue

            # Create a mask for the current contour
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)

            has_children = hierarchy[0][i][2] != -1
            children_too_small = False
            if has_children:
                child_idx = hierarchy[0][i][2]
                while child_idx != -1:
                    # Subtract the child contour from the parent's mask
                    child = contours[child_idx]
                    if cv2.contourArea(child) >= min_area:
                        children_too_small = True
                        cv2.drawContours(contour_mask, [child], -1, 0, -1)
                    child_idx = hierarchy[0][child_idx][0]
            has_children = has_children and not children_too_small

            # Find the largest inscribed circle in the contour mask
            dist_map = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
            _, radius, _, center = cv2.minMaxLoc(dist_map)

            if cv2.pointPolygonTest(contour, center, False) >= 0:
                yield contour, (center, radius), number
            else:
                print('dafuq')


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR_BGR)
    palette = load_palette()
    palette_dict = {color: i for i, color in enumerate(palette) if True or i in [0, 1, 2, 3, 4]}

    image = apply_median_filter(image, kernel_size=13)
    # image = unsharp_mask(image, amount=1.5)
    print('Preprocessed')
    quantized_image = quantize_colors(image, palette).astype('uint8')
    print('Quantized')
    contour_image = np.full_like(quantized_image, 255)
    image = draw_color_contours(quantized_image, contour_image, palette_dict, 50)
    print('Generated contours')

    cv2.imwrite("output_quantized.png", quantized_image)
    cv2.imwrite("output_contours.png", image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    process_image("test3.jpg")
