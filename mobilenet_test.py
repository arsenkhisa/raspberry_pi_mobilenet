import cv2
import numpy as np

MODEL = "mobilenetv2.onnx"
LABELS = "imagenet_classes.txt"
IMAGE = "cat.jpg"   # можешь поставить своё имя файла

IMG_SIZE = 224


def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Image not found: {img_path}")

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # MobileNet требует нормализации [0..1] и RGB
    blob = cv2.dnn.blobFromImage(
        img_resized,
        scalefactor=1.0 / 255.0,
        size=(IMG_SIZE, IMG_SIZE),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )
    return img, blob


def main():
    print("[*] Loading model...")
    net = cv2.dnn.readNetFromONNX(MODEL)

    print("[*] Loading labels...")
    labels = load_labels(LABELS)

    print("[*] Preprocessing image...")
    img, blob = preprocess(IMAGE)
    net.setInput(blob)

    print("[*] Running inference...")
    out = net.forward()[0]

    # softmax (не обязательно, но удобно для читаемости)
    exp = np.exp(out - np.max(out))
    probs = exp / exp.sum()

    # top-5 классов
    top5 = probs.argsort()[-5:][::-1]

    print("\nTop-5 predictions:")
    for idx in top5:
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"{idx:4d}: {label:30s} {probs[idx]:.4f}")

    # Сохраняем изображение с подписью
    best_idx = top5[0]
    best_label = labels[best_idx]
    cv2.putText(
        img, best_label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
    )
    cv2.imwrite("result.jpg", img)
    print("\nSaved result as result.jpg\n")


if __name__ == "__main__":
    main()
