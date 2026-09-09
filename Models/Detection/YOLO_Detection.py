from ultralytics import YOLO
import cv2
import os
import random
import numpy as np
from collections import Counter


def draw_boxes(image, results, class_names, color_sample):
    image_with_boxes = image.copy()

    for box in results.boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = f"{class_names[cls_id]}"
        color = random.choice(color_sample)

        cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image_with_boxes,
            label,
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    return image_with_boxes


def detection(image, class_names, color_sample, weights):
    weights_path = os.path.join(os.path.dirname(__file__), "..", "Weights", f"{weights}")
    model = YOLO(weights_path)

    results = model(image)[0]

    image_numpy = np.array(image)
    image_with_boxes = draw_boxes(image_numpy, results, class_names, color_sample)

    # Count classes as a list aligned with class_names
    filtered_labels = [int(box.cls[0]) for box in results.boxes]
    counter = Counter(filtered_labels)
    counter_list = [counter[i] for i in range(len(class_names))]

    # Confidence scores
    scores = [float(box.conf[0]) for box in results.boxes]

    return image_with_boxes, counter_list, scores
