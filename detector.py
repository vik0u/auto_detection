import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import os

model = YOLO('yolo11n.pt')

video_path = "cvtest.avi"
cap = cv2.VideoCapture(video_path)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def detect_object(frame_for_detection, frame_for_drawing):
    """
    Функция детекции объекта и отрисовки бокса
    :param frame_for_detection: np.ndarray маскированный кадр из потока
    :param frame_for_drawing: np.ndarray кадр из потока без маски для отрисовки
    :return: tuple (bool, np.ndarray)
        - transport_detected (bool) — флаг наличия обнаруженных объектов заданных классов,
        - frame (np.ndarray) — кадр с отрисованными bounding боксаами обнаруженных объектов.
    """
    results = model.predict(frame_for_detection, classes=config['classes'], conf=config['confidence_threshold'], iou = config['nms_threshold'])

    # Проверяем наличие детекций
    transport_detected = any(len(result.boxes) > 0 for result in results)

    # Отрисовываем боксы
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_for_drawing, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return transport_detected, frame_for_drawing

def apply_roi_mask(frame):
    """
    Маскируем кадр для избежания ложной детекции проезжающих машин

    :param frame: np.ndarray кадр из потока
    :return: tuple (np.ndarray, np.ndarray)
        - masked_frame (np.ndarray) — маскированный кадр,
        - mask (np.ndarray) — бинарная маска
    """
    height, width = frame.shape[:2]
    roi_vertices = np.array(config['roi_vertices'])

    # Добавляем нижнюю точку с Y=высота кадра, X равен X нижнего левого угла roi_vertices[1]
    bottom_point = [roi_vertices[1][0], height]

    roi_vertices_with_bottom = np.array([
        roi_vertices[0],          # верхний левый
        roi_vertices[1],          # нижний левый
        bottom_point,             # нижняя точка по нижней границе кадра
        roi_vertices[2],          # нижний правый
        roi_vertices[3],          # верхний правый
    ])

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roi_vertices_with_bottom], (255, 255, 255))
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame

# Создаём папку для сохранения, если её нет
output_dir = 'detected_frames'
os.makedirs(output_dir, exist_ok=True)

frame_counter = 0  # счётчик кадров
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    masked_frame = apply_roi_mask(frame)
    transport_found, processed_frame = detect_object(masked_frame, frame.copy())

    status = "Транспорт обнаружен" if transport_found else "Транспорта нет"
    print(status)
    #можно добавить отрисовку бокса и вывод
    #cv2.putText(processed_frame, status, (10, 30),
    #           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #cv2.imshow('Vehicle Detection', processed_frame)

    # Сохраняем кадр, если транспорт обнаружен
    if transport_found:
        filename = os.path.join(output_dir, f'detected_{frame_counter:05d}.jpg')
        cv2.imwrite(filename, processed_frame)

    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()