import cv2
from ultralytics import YOLO


# Функция для отображения результатов
def draw_boxes(frame, detections):
    for detection in detections:
        x, y, w, h = detection['bbox']
        center_x, center_y = int(x + w / 2), int(y + h / 2)
        color = (0, 255, 0)  # Зеленый по умолчанию

        if detection['class'] == 'military_vehicle':
            color = (0, 0, 255)  # Красный
        elif detection['class'] == 'civil_vehicle':
            color = (0, 255, 0)  # Зеленый
        elif detection['class'] == 'person':
            color = (0, 255, 255)  # Желтый

        # Рисуем прямоугольник и крестик
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.drawMarker(frame, (center_x, center_y), color, cv2.MARKER_CROSS)

        # Подпись класса объекта
        cv2.putText(frame, detection['class'], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    # Настройка модели YOLOv8
    yolov8 = YOLO(weights="best.pt")

    # Получаем доступ к веб-камере
    cap = cv2.VideoCapture('rtsp://192.168.144.25:8554/main.264')

    # Основной цикл
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Запускаем детектирование
        detections = yolov8.detect(frame)

        # Рисуем результаты детектирования
        draw_boxes(frame, detections)

        # Отображаем кадр
        cv2.imshow('Обнаружение в реальном времени', frame)

        # Остановка цикла по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

