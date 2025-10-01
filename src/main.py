import os
import cv2, numpy

RESOURCES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources")
)
BING_IMAGE = os.path.join(RESOURCES_DIR, "bing.png")


def detect_light_buildings(image: cv2.typing.MatLike):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured = cv2.bilateralFilter(gray, 7, 35, 35)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(blured)
    _, threshold = cv2.threshold(clahe_applied, 210, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low1 = numpy.array([0, 80, 80])
    high1 = numpy.array([10, 255, 255])
    low2 = numpy.array([160, 80, 80])
    high2 = numpy.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, low1, high1)
    mask2 = cv2.inRange(hsv, low2, high2)
    mask_color = cv2.bitwise_or(mask1, mask2)

    combined = cv2.bitwise_or(mask_color, threshold)

    kernel = numpy.ones((6, 6), numpy.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            accepted.append(contour)
    return accepted


def main():
    original = cv2.imread(BING_IMAGE, cv2.IMREAD_COLOR)
    contours = detect_light_buildings(original)
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        cv2.drawContours(original, [box], 0, (0, 255, 0), 2)

    cv2.imshow("Original", original)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
