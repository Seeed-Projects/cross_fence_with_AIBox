import cv2
import numpy as np
from common.toolbox import id_to_color

line_points = []  # Store two clicked points for fence line
# Global tracker to store previous sides for track IDs
track_prev_sides = {}  # track_id: side (+/-)

# Define which side is "inside" (e.g. positive or negative)
FENCE_INNER_IS_NEGATIVE = True
track_crossed_once = set()


def get_fence_line_from_user(frame):
    clone = frame.copy()
    window_name = "Select Fence Line - Click Two Points"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(line_points) < 2:
                line_points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while len(line_points) < 2:
        disp = clone.copy()
        for pt in line_points:
            cv2.circle(disp, pt, 5, (0, 0, 255), -1)
        if len(line_points) == 2:
            cv2.line(disp, line_points[0], line_points[1], (255, 0, 0), 2)
        cv2.imshow(window_name, disp)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyWindow(window_name)
    return line_points if len(line_points) == 2 else None

def inference_result_handler(original_frame, infer_results, labels, config_data, tracker=None):
    if len(line_points) < 2:
        pts = get_fence_line_from_user(original_frame)
        if not pts:
            return original_frame

    cv2.line(original_frame, line_points[0], line_points[1], (255, 0, 0), 4)
    cv2.putText(original_frame, "Fence Line", (line_points[0][0] + 5, line_points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    detections = extract_detections(original_frame, infer_results, config_data)
    frame_with_detections = draw_detections(detections, original_frame, labels, tracker=tracker, line_pts=line_points)
    return frame_with_detections

def draw_detection(image: np.ndarray, box: list, labels: list, score: float, color: tuple, track=False):
    ymin, xmin, ymax, xmax = map(int, box)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    top_text = f"{labels[0]}: {score:.1f}%" if not track or len(labels) == 2 else f"{score:.1f}%"
    bottom_text = labels[1] if track and len(labels) == 2 else labels[0] if track else None

    text_color = (255, 255, 255)
    border_color = (0, 0, 0)

    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, border_color, 2, cv2.LINE_AA)
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, text_color, 1, cv2.LINE_AA)

    if bottom_text:
        pos = (xmax - 50, ymax - 6)
        cv2.putText(image, bottom_text, pos, font, 0.5, border_color, 2, cv2.LINE_AA)
        cv2.putText(image, bottom_text, pos, font, 0.5, text_color, 1, cv2.LINE_AA)
        
def point_side(pt, line):
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)


def draw_detections(detections: dict, img_out: np.ndarray, labels, tracker=None, line_pts=None):
    boxes = detections["detection_boxes"]
    scores = detections["detection_scores"]
    num_detections = detections["num_detections"]
    classes = detections["detection_classes"]

    if tracker:
        dets_for_tracker = [[*boxes[idx], scores[idx]] for idx in range(num_detections)]
        if not dets_for_tracker:
            return img_out

        online_targets = tracker.update(np.array(dets_for_tracker))

        for track in online_targets:
            track_id = track.track_id
            ymin, xmin, ymax, xmax = map(int, track.tlbr)
            mid_point = ((xmin + xmax) // 2, (ymin + ymax) // 2)

            current_side = point_side(mid_point, line_pts) if line_pts else 0
            previous_side = track_prev_sides.get(track_id)

            is_crossing_now = False
            if line_pts and previous_side is not None and track_id not in track_crossed_once:
                crossed = (current_side * previous_side < 0)
                into_inner = (current_side < 0) if FENCE_INNER_IS_NEGATIVE else (current_side > 0)
                if crossed and into_inner:
                    is_crossing_now = True
                    track_crossed_once.add(track_id)  # Only mark once

            track_prev_sides[track_id] = current_side

            is_already_crossed = track_id in track_crossed_once
            color = (0, 0, 255) if is_crossing_now or is_already_crossed else (0, 255, 0)
            best_idx = find_best_matching_detection_index(track.tlbr, boxes)
            label = [f"ID {track_id}"] if best_idx is None else [labels[classes[best_idx]], f"ID {track_id}"]
            draw_detection(img_out, [ymin, xmin, ymax, xmax], label, track.score * 100.0, color, track=True)

    else:
        for idx in range(num_detections):
            ymin, xmin, ymax, xmax = map(int, boxes[idx])
            mid_point = ((xmin + xmax) // 2, (ymin + ymax) // 2)
            is_crossing = False
            if line_pts:
                current_side = point_side(mid_point, line_pts)
                is_crossing = current_side < 0 if FENCE_INNER_IS_NEGATIVE else current_side > 0
                if is_crossing:
                    track_crossed_once.add(idx)

            is_already_crossed = idx in track_crossed_once
            color = (0, 0, 255) if is_crossing or is_already_crossed else tuple(id_to_color(classes[idx]).tolist())
            draw_detection(img_out, boxes[idx], [labels[classes[idx]]], scores[idx] * 100.0, color)

    if track_crossed_once:
        cv2.putText(img_out, "Warning: someone is crossing the fence",
                    (20, img_out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return img_out


def denormalize_and_rm_pad(box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
    for i, x in enumerate(box):
        box[i] = int(x * size)
        if (input_width != size) and (i % 2 != 0):
            box[i] -= padding_length
        if (input_height != size) and (i % 2 == 0):
            box[i] -= padding_length
    return box

def extract_detections(image: np.ndarray, detections: list, config_data) -> dict:
    visualization_params = config_data["visualization_params"]
    score_threshold = visualization_params.get("score_thres", 0.8)
    max_boxes = visualization_params.get("max_boxes_to_draw", 50)

    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []

    for class_id, detection in enumerate(detections):
        for det in detection:
            bbox, score = det[:4], det[4]
            if score >= 0.6 and class_id == 0:
                denorm_bbox = denormalize_and_rm_pad(bbox, size, padding_length, img_height, img_width)
                all_detections.append((score, class_id, denorm_bbox))

    all_detections.sort(reverse=True, key=lambda x: x[0])
    top_detections = all_detections[:max_boxes]

    scores, class_ids, boxes = zip(*top_detections) if top_detections else ([], [], [])

    return {
        'detection_boxes': list(boxes),
        'detection_classes': list(class_ids),
        'detection_scores': list(scores),
        'num_detections': len(top_detections)
    }

def find_best_matching_detection_index(track_box, detection_boxes):
    best_iou = 0
    best_idx = -1
    for i, det_box in enumerate(detection_boxes):
        iou = compute_iou(track_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx if best_idx != -1 else None

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-5)