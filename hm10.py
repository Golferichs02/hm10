import cv2 as cv
import numpy as np
import argparse

def user_interaction():
    parser = argparse.ArgumentParser(description='Real-time Harris Corner Detection in Video')
    parser.add_argument('-v', '--video',
                        type=str,
                        required=True,
                        help="Path to the video file or '0' for camera input")
    parser.add_argument('-i', '--image',
                        type=str,
                        required=True,
                        help="Path to the reference image")
    parser.add_argument('--resize',
                        type=int,
                        required=True,
                        help="Percentage to resize the frames and image")
    return parser.parse_args()

def load_image(path):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}. Please check the path.")
    return img

def resize_image(img, resize_percent):
    height, width = img.shape[:2]
    dimensions = (int(width * resize_percent / 100), int(height * resize_percent / 100))
    resized_img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    return resized_img

def extract_corners_and_descriptors(image):
    # Aplicar un filtro Gaussiano para suavizar la imagen
    image = cv.GaussianBlur(image, (7, 7), 0)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    harris_corners = cv.cornerHarris(gray, 15, 11, 0.04)
    threshold = 0.10 * harris_corners.max()  # Aumentar el umbral
    corner_positions = np.argwhere(harris_corners > threshold)
    keypoints = [cv.KeyPoint(float(x[1]), float(x[0]), 3) for x in corner_positions]
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints, descriptors = brief.compute(gray, keypoints)
    return keypoints, descriptors

def calculate_centroid(keypoints):
    if not keypoints:
        return None
    x = np.mean([kp.pt[0] for kp in keypoints])
    y = np.mean([kp.pt[1] for kp in keypoints])
    return (x, y)

def determine_quadrant(centroid, width, height):
    mid_x = width / 2
    m = -height / width  # pendiente 
    b = height  # intercepta en el borde superior derecho (y = mx + b para x = 0)
    x, y = centroid
    diagonal_line_y = m * x + b
    if x > mid_x and y < diagonal_line_y:
        return 1  # Cuadrante superior derecho
    elif x < mid_x and y < diagonal_line_y:
        return 2  # Cuadrante superior izquierdo
    elif x < mid_x and y > diagonal_line_y:
        return 3  # Cuadrante inferior izquierdo
    else:
        return 4  # Cuadrante inferior derecho

def draw_frame_around_centroid(frame, centroid, frame_size):
    """Dibuja un marco cuadrado alrededor del centroide."""
    x, y = int(centroid[0]), int(centroid[1])
    top_left = (x - frame_size // 2, y - frame_size // 2)
    bottom_right = (x + frame_size // 2, y + frame_size // 2)
    cv.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)

def draw_matches(img1, kp1, des1, img2, kp2, des2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.80 * n.distance]

    if len(good_matches) >= 4:  # Asegúrate de tener al menos 4 buenos matches
        # Extraer las coordenadas de los buenos matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimar la homografía con RANSAC
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 20.0)
        matchesMask = mask.ravel().tolist()

        # Dibujar solo los matches que son inliers
        draw_params = dict(matchColor = (0,255,0),  # Dibuja matches en verde
                           singlePointColor = None,
                           matchesMask = matchesMask,  # Dibuja solo inliers
                           flags = 2)
        matched_img = cv.drawMatches(img1, kp1, img2, kp2, good_matches, 
                                     None, **draw_params)
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 4))
        matchesMask = None
        matched_img = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None)
        
    return matched_img

def print_time_and_counts(quadrant_times, fps):
    time_in_seconds = {k: {'time': v['time'] / fps, 'count': v['count']} 
                       for k, v in quadrant_times.items()}
    print("Time spent and entry counts in quadrants:", time_in_seconds)

def initialize_capture(video_path):
    cap = cv.VideoCapture(video_path if video_path != "0" else 0)
    fps = cap.get(cv.CAP_PROP_FPS)
    return cap, fps

def calculate_black_centroid_hsv(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    mask = cv.inRange(hsv, lower_black, upper_black)
    moments = cv.moments(mask)
    if moments["m00"] != 0:
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
        return (x, y)
    else:
        return None

def process_frame(frame, resize_percent, reference_img_resized, ref_kp, 
                  ref_des, last_quadrant, quadrant_times, previous_centroid):
    frame_resized = resize_image(frame, resize_percent)
    kp, des = extract_corners_and_descriptors(frame_resized)
    
    # Definir frame_size y max_distance adaptativos
    frame_size = int(0.25 * min(frame_resized.shape[:2]))
    max_distance = int(0.05 * min(frame_resized.shape[:2]))

    # Calcular centroide HSV para el color negro
    hsv_centroid = calculate_black_centroid_hsv(frame_resized)
    if kp:
        current_centroid = calculate_centroid(kp)
        if current_centroid and hsv_centroid:
            # Calcular el promedio de las posiciones
            averaged_centroid = ((current_centroid[0] + hsv_centroid[0]) / 2, 
                                 (current_centroid[1] + hsv_centroid[1]) / 2)
        else:
            averaged_centroid = current_centroid or hsv_centroid

        if averaged_centroid:
            limited_centroid = limit_centroid_movement(averaged_centroid, 
                                                       previous_centroid, 
                                                       max_distance)
            previous_centroid = limited_centroid  # Actualizar el último centroide
            x, y = int(limited_centroid[0]), int(limited_centroid[1])
            cv.circle(frame_resized, (x, y), 5, (255, 0, 0), -1)
            draw_frame_around_centroid(frame_resized, 
                                       limited_centroid, 
                                       frame_size)  # Dibuja un marco alrededor del centroide
            
            quadrant = determine_quadrant(limited_centroid, 
                                          frame_resized.shape[1], 
                                          frame_resized.shape[0])
            if quadrant != last_quadrant:
                quadrant_times[quadrant]['count'] += 1
                last_quadrant = quadrant
            quadrant_times[quadrant]['time'] += 1

    draw_lines(frame_resized)
    matched_img = draw_matches(frame_resized, kp, des, reference_img_resized, 
                               ref_kp, ref_des)
    return matched_img, last_quadrant, previous_centroid

def draw_lines(frame):
    height, width = frame.shape[:2]
    cv.line(frame, (0, height), (width, 0), (0, 255, 0), 2)
    cv.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 0), 2)

def limit_centroid_movement(current_centroid, previous_centroid, max_distance):
    if previous_centroid is None or current_centroid is None:
        return current_centroid
    distance = np.sqrt((current_centroid[0] - previous_centroid[0])**2 
                       + (current_centroid[1] - previous_centroid[1])**2)
    if distance > max_distance:
        direction = ((current_centroid[0] - previous_centroid[0]), 
                     (current_centroid[1] - previous_centroid[1]))
        norm_direction = (direction[0] / distance, direction[1] / distance)
        limited_centroid = (previous_centroid[0] + norm_direction[0] * max_distance, 
                            previous_centroid[1] + norm_direction[1] * max_distance)
        return limited_centroid
    return current_centroid

def main_loop(video_path, reference_img, resize_percent):
    cap, fps = initialize_capture(video_path)
    reference_img_resized = resize_image(reference_img, resize_percent)
    ref_kp, ref_des = extract_corners_and_descriptors(reference_img_resized)
    quadrant_times = {1: {'time': 0, 'count': 0}, 2: {'time': 0, 'count': 0}, 3: {'time': 0, 'count': 0}, 4: {'time': 0, 'count': 0}}
    last_quadrant = None
    previous_centroid = None  # Guardar el último centroide para la restricción de movimiento

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        matched_img, last_quadrant, previous_centroid = process_frame(frame, resize_percent,
                                                                       reference_img_resized, 
                                                                       ref_kp, ref_des, 
                                                                       last_quadrant, 
                                                                       quadrant_times, 
                                                                       previous_centroid)
        cv.imshow('Real-time Feature Matching', matched_img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    print_time_and_counts(quadrant_times, fps)

def run_pipeline():
    args = user_interaction()
    reference_img = load_image(args.image)
    main_loop(args.video, reference_img, args.resize)

if __name__ == "__main__":
    run_pipeline()