import cv2


def load():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 100000
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    return cv2.SimpleBlobDetector_create(params)


def infer(model, img):
    # model is a cv2.SimpleBlobDetector
    keypoints = model.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    dets = []
    for k in keypoints:
        x, y = k.pt
        s = k.size
        dets.append({"x": float(x), "y": float(y), "w": float(s), "h": float(s), "angle": 0.0})
    return dets