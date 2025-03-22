import cv2
import numpy as np
from skimage.segmentation import watershed
from scipy.ndimage import label, center_of_mass
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
    return quantized, indices.reshape(image.shape[:2])


def segment_image(image, palette):
    markers_total = []

    # palette = [palette[i] for i in [4, 16]]

    for i, color in enumerate(palette):
        solid_color = np.full_like(image, color)
        parts = cv2.absdiff(image, solid_color)

        gray = cv2.cvtColor(parts, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)

        # deal with small areas (calculate closing of image)
        # thresh = cv2.dilate(thresh, None, iterations=4)
        # thresh = cv2.erode(thresh, None, iterations=4)
        thresh = apply_median_filter(thresh, kernel_size=7)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        _, markers = cv2.connectedComponents(thresh)

        markers_total.append((watershed(-gray, markers, mask=thresh), contours))

    return markers_total


def draw_labels(image, labels: np.ndarray, indices, contours, contours_color=(0, 255, 0), text_color=(255, 0, 0)):
    area_threshold = 50

    deleted_contours = [contour for contour in contours if cv2.contourArea(contour) < area_threshold]
    reduced_contours = [contour for contour in contours if cv2.contourArea(contour) >= area_threshold]

    cv2.drawContours(image, reduced_contours, -1, contours_color, 1)

    labeled, num_features = label(labels)

    for i in range(1, num_features + 1):
        mask: np.ndarray = (labeled == i).astype(np.uint8)
        moments = cv2.moments(mask)

        if moments['m00'] == 0:
            continue

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        text_x = cx
        text_y = cy

        # Проверяем, не попали ли мы во вложенный контур
        center_color = indices[cy, cx]
        contour_point = contours[0][:, 0][0]  # Точка для проверки цвета
        actual_color = indices[contour_point[1], contour_point[0]]
        # if center_color == actual_color:
        #     print("match")

        # Проверяем, находится ли центр масс внутри контура
        in_contour = (any(cv2.pointPolygonTest(contour, (cx, cy), False) >= 0 for contour in contours)
                      and center_color == actual_color)

        offset = 5

        if not in_contour:
            # Находим ближайшую точку контура к центру масс
            best_dist = float('inf')
            best_point = (cx, cy)
            for contour in contours:
                for point in contour[:, 0]:
                    px, py = point
                    dist = (cx - px) ** 2 + (cy - py) ** 2  # Евклидово расстояние в квадрате
                    if dist < best_dist:
                        best_dist = dist
                        best_point = (px, py)

            bx, by = best_point

            # Вектор от центра масс к ближайшей точке контура
            vec_x = bx - cx
            vec_y = by - cy
            length = np.hypot(vec_x, vec_y)

            if length > 0:
                vec_x = int(vec_x / length * offset)
                vec_y = int(vec_y / length * offset)

            text_x = bx + vec_x
            text_y = by + vec_y

        in_deleted_contour = any(cv2.pointPolygonTest(contour, (float(text_x), float(text_y)), False) >= 0 for contour in deleted_contours)
        if in_deleted_contour:
            continue

        # draw single characters to configure spacing
        color_str = str(actual_color)
        text = [str(c) for c in color_str]

        fontFace = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        thickness = 1
        text_width, text_height = cv2.getTextSize(color_str, fontFace, fontScale, thickness)[0]
        center_coordinates = (text_x - int(text_width / 2), text_y + int(text_height / 2))
        spacing = 5.5

        for i, char in enumerate(text):
            coords_x, coords_y = center_coordinates
            cv2.putText(image, char, (int(coords_x + i * spacing), int(coords_y)), fontFace, fontScale, text_color, thickness, cv2.LINE_AA)

    return image


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


def process_image(image_path, num_colors=20):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR_BGR)
    image = apply_median_filter(image, kernel_size=13)
    # image = unsharp_mask(image, amount=1.5)

    # palette = extract_palette(image, num_colors)
    palette = load_palette()

    image, indices = quantize_colors(image, palette)

    image = image.astype(np.uint8)

    cv2.imwrite("output_unmarked.png", image)

    contour_image = np.full_like(image, 255)

    for i, (color_labels, contours) in enumerate(segment_image(image, palette)):
        contour_image = draw_labels(contour_image, color_labels, indices, contours, contours_color=(200, 200, 200), text_color=(0, 0, 0))
        image = draw_labels(image, color_labels, indices, contours)

    cv2.imwrite("output.png", contour_image)
    cv2.imwrite("output_full.png", image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    process_image("test3.jpg")
