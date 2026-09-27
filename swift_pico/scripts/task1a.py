#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import os


REQUIRED_MARKER_IDS = [80, 85, 90, 95]

OUTPUT_SIZE = 900

GRID_COLUMNS = "ABCDEFGHIJK"
GRID_SIZE = 11

# The arena contains 12 x 12 equal cells.
# Therefore the 11 x 11 interior intersections
# are at 1/12, 2/12, ... 11/12 of the arena.
GRID_STEP = OUTPUT_SIZE / 12.0


def detect_aruco_markers(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    aruco_dict = cv2.aruco.Dictionary_get(
        cv2.aruco.DICT_4X4_250
    )

    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    markers = {}

    if ids is not None:

        for marker_corners, marker_id in zip(
            corners,
            ids.flatten()
        ):

            marker_id = int(marker_id)

            if marker_id in REQUIRED_MARKER_IDS:
                markers[marker_id] = marker_corners[0]

    return markers


def get_marker_centers(markers):

    centers = {}

    for marker_id, corners in markers.items():

        centers[marker_id] = np.mean(
            corners,
            axis=0
        )

    return centers


def perspective_transform(image, centers):

    src_points = np.float32([
        centers[80],
        centers[85],
        centers[90],
        centers[95]
    ])

    dst_points = np.float32([
        [0, 0],
        [OUTPUT_SIZE - 1, 0],
        [OUTPUT_SIZE - 1, OUTPUT_SIZE - 1],
        [0, OUTPUT_SIZE - 1]
    ])

    matrix = cv2.getPerspectiveTransform(
        src_points,
        dst_points
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (OUTPUT_SIZE, OUTPUT_SIZE)
    )

    return warped


def get_contour_centers(mask, min_area):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        detections.append(
            (cx, cy, area)
        )

    return detections


def detect_survivors(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Yellow survivor
    yellow_lower = np.array(
        [20, 100, 100]
    )

    yellow_upper = np.array(
        [40, 255, 255]
    )

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    # Red survivor
    red_lower1 = np.array(
        [0, 100, 100]
    )

    red_upper1 = np.array(
        [10, 255, 255]
    )

    red_lower2 = np.array(
        [170, 100, 100]
    )

    red_upper2 = np.array(
        [180, 255, 255]
    )

    red_mask1 = cv2.inRange(
        hsv,
        red_lower1,
        red_upper1
    )

    red_mask2 = cv2.inRange(
        hsv,
        red_lower2,
        red_upper2
    )

    red_mask = cv2.bitwise_or(
        red_mask1,
        red_mask2
    )

    yellow = get_contour_centers(
        yellow_mask,
        500
    )

    red = get_contour_centers(
        red_mask,
        500
    )

    return yellow, red


def get_grid_intersections():

    intersections = []

    for row in range(GRID_SIZE):

        for col in range(GRID_SIZE):

            x = (col + 1) * GRID_STEP
            y = (row + 1) * GRID_STEP

            label = (
                GRID_COLUMNS[col]
                + str(row + 1)
            )

            intersections.append(
                (label, x, y)
            )

    return intersections


def nearest_grid_intersection(x, y):

    intersections = get_grid_intersections()

    best_label = None
    best_distance = float("inf")

    for label, gx, gy in intersections:

        distance = (
            (x - gx) ** 2
            + (y - gy) ** 2
        )

        if distance < best_distance:

            best_distance = distance
            best_label = label

    return best_label


def write_results(
    image_path,
    marker_ids,
    critical_survivors,
    stable_survivors
):

    directory = os.path.dirname(
        os.path.abspath(image_path)
    )

    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        directory,
        filename + "_results.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Detected marker IDs: {marker_ids}\n"
        )

        file.write("\n")

        file.write(
            "Critical Survivors: "
            + ", ".join(critical_survivors)
            + "\n"
        )

        file.write(
            "Stable Survivors: "
            + ", ".join(stable_survivors)
            + "\n"
        )

    return output_path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True
    )

    args = parser.parse_args()

    image = cv2.imread(
        args.image
    )

    if image is None:

        raise RuntimeError(
            f"Could not load image: {args.image}"
        )

    # Step 1: ArUco markers

    markers = detect_aruco_markers(
        image
    )

    marker_ids = sorted(
        markers.keys()
    )

    print(
        "Detected marker IDs:",
        marker_ids
    )

    if marker_ids != REQUIRED_MARKER_IDS:

        raise RuntimeError(
            "Required ArUco markers "
            "80, 85, 90 and 95 were not "
            "all detected."
        )

    # Step 2: Perspective transform

    centers = get_marker_centers(
        markers
    )

    warped = perspective_transform(
        image,
        centers
    )

    cv2.imwrite(
        "arena_warped.jpg",
        warped
    )

    print(
        "Perspective transformation complete."
    )

    print(
        "Saved: arena_warped.jpg"
    )

    # Step 3: Detect survivors

    yellow_survivors, red_survivors = (
        detect_survivors(warped)
    )

    print(
        "Yellow survivors detected:",
        len(yellow_survivors)
    )

    print(
        "Red survivors detected:",
        len(red_survivors)
    )

    # Step 4: Map survivors to intersections

    stable_survivors = []

    for x, y, area in yellow_survivors:

        label = nearest_grid_intersection(
            x,
            y
        )

        stable_survivors.append(
            label
        )

        print(
            f"Yellow: ({x:.1f}, {y:.1f}) -> {label}"
        )

    critical_survivors = []

    for x, y, area in red_survivors:

        label = nearest_grid_intersection(
            x,
            y
        )

        critical_survivors.append(
            label
        )

        print(
            f"Red: ({x:.1f}, {y:.1f}) -> {label}"
        )

    # Remove duplicates and sort

    stable_survivors = sorted(
        set(stable_survivors)
    )

    critical_survivors = sorted(
        set(critical_survivors)
    )

    # Step 5: Write required results

    output_path = write_results(
        args.image,
        marker_ids,
        critical_survivors,
        stable_survivors
    )

    print(
        "Results saved:",
        output_path
    )


if __name__ == "__main__":
    main()