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
        return [tuple(int(line[i:i + 2], 16) for i in (4, 2, 0)) for line in file.read().splitlines() if not line.__contains__('#')]


def apply_median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)


def quantize_colors(image, palette):
    reshaped = image.reshape(-1, 3)
    tree = KDTree(palette)
    _, indices = tree.query(reshaped)
    quantized = np.array(palette)[indices].reshape(image.shape)
    return quantized, indices.reshape(image.shape[:2])


def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, markers = cv2.connectedComponents(thresh)
    return watershed(-gray, markers, mask=thresh)


def draw_labels(image, labels, indices):
    labeled, num_features = label(labels)
    centers = center_of_mass(labels, labeled, range(1, num_features + 1))

    for i, (cy, cx) in enumerate(centers):
        if np.isnan(cx) or np.isnan(cy):
            continue
        text = str(indices[int(cy), int(cx)])
        cv2.putText(image, text, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    return image


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
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
    image = apply_median_filter(image, kernel_size=11)
    image = unsharp_mask(image, amount=1.5)

    # palette = extract_palette(image, num_colors)
    palette = load_palette()

    image, indices = quantize_colors(image, palette)

    image = image.astype(np.uint8)

    # labels = segment_image(image)
    # image = draw_labels(image, labels, indices)

    cv2.imwrite("output.png", image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    print("Processing complete. Output saved as output.png")


if __name__ == "__main__":
    process_image("test1.jpg")
