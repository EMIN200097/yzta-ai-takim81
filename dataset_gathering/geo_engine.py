import cv2
import numpy as np
import math


def angle_between_points_cv2(A, B, C):
    """
    Üç nokta arasındaki açıyı hesaplar (B merkez nokta)
    OpenCV'nin fastAtan2 fonksiyonunu kullanarak hızlı hesaplama

    Args:
        A, B, C: (x, y) koordinat tuple'ları veya landmark objeleri

    Returns:
        float: Açı değeri (derece cinsinden, 0-180 arası)
    """
    try:
        # Eğer MediaPipe landmark objesi ise koordinatları çıkar
        if hasattr(A, 'x') and hasattr(A, 'y'):
            A = (A.x, A.y)
        if hasattr(B, 'x') and hasattr(B, 'y'):
            B = (B.x, B.y)
        if hasattr(C, 'x') and hasattr(C, 'y'):
            C = (C.x, C.y)

        # Numpy array'e çevir ve float32'ye cast et
        A = np.array(A, dtype=np.float32)
        B = np.array(B, dtype=np.float32)
        C = np.array(C, dtype=np.float32)

        # Vektörleri hesapla
        BA = A - B
        BC = C - B

        # Sıfır vektör kontrolü
        if np.allclose(BA, 0, atol=1e-8) or np.allclose(BC, 0, atol=1e-8):
            return 0.0

        # OpenCV'nin hızlı atan2 fonksiyonu ile açıları hesapla
        # Parametreleri açıkça float32 olarak cast et
        angle1 = cv2.fastAtan2(float(BA[1]), float(BA[0]))
        angle2 = cv2.fastAtan2(float(BC[1]), float(BC[0]))

        # Açı farkını hesapla
        angle = abs(angle2 - angle1)

        # 0-180 arası normalize et
        if angle > 180:
            angle = 360 - angle

        return float(angle)

    except Exception as e:
        print(f"⚠️ CV2 açı hesaplama hatası: {e}")
        # Hata durumunda numpy yöntemini dene
        return angle_between_points_numpy(A, B, C)


def angle_between_points_numpy(A, B, C):
    """
    Alternatif açı hesaplama yöntemi - NumPy ile dot product kullanarak
    Daha hassas ama biraz daha yavaş
    """
    try:
        # Eğer MediaPipe landmark objesi ise koordinatları çıkar
        if hasattr(A, 'x') and hasattr(A, 'y'):
            A = (A.x, A.y)
        if hasattr(B, 'x') and hasattr(B, 'y'):
            B = (B.x, B.y)
        if hasattr(C, 'x') and hasattr(C, 'y'):
            C = (C.x, C.y)

        A = np.array(A, dtype=np.float64)
        B = np.array(B, dtype=np.float64)
        C = np.array(C, dtype=np.float64)

        # Vektörleri hesapla
        BA = A - B
        BC = C - B

        # Sıfır vektör kontrolü
        norm_BA = np.linalg.norm(BA)
        norm_BC = np.linalg.norm(BC)

        if norm_BA < 1e-8 or norm_BC < 1e-8:
            return 0.0

        # Dot product ile açı hesapla
        cos_angle = np.dot(BA, BC) / (norm_BA * norm_BC)

        # Numerical stability için clamp
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Açıyı dereceye çevir
        angle = np.arccos(cos_angle) * 180.0 / np.pi

        return float(angle)

    except Exception as e:
        print(f"⚠️ NumPy açı hesaplama hatası: {e}")
        return 0.0


def calculate_body_angles(landmarks, side='auto'):
    """
    Vücut landmark'larından önemli açıları hesaplar

    Args:
        landmarks: MediaPipe pose landmarks
        side: 'left', 'right', veya 'auto' (en iyi tarafı seç)

    Returns:
        dict: Hesaplanan açılar
    """
    try:
        # MediaPipe pose landmark indeksleri
        pose_indices = {
            'left': {
                'ear': 7, 'shoulder': 11, 'elbow': 13, 'wrist': 15,
                'hip': 23, 'knee': 25, 'ankle': 27, 'foot': 31
            },
            'right': {
                'ear': 8, 'shoulder': 12, 'elbow': 14, 'wrist': 16,
                'hip': 24, 'knee': 26, 'ankle': 28, 'foot': 32
            }
        }

        # Otomatik taraf seçimi
        if side == 'auto':
            left_vis = sum(landmarks[i].visibility for i in [7, 11, 23, 25, 27] if i < len(landmarks))
            right_vis = sum(landmarks[i].visibility for i in [8, 12, 24, 26, 28] if i < len(landmarks))
            side = 'left' if left_vis > right_vis else 'right'

        indices = pose_indices[side]
        angles = {}

        # Omuz açısı (kulak-omuz-kalça)
        if all(idx < len(landmarks) for idx in [indices['ear'], indices['shoulder'], indices['hip']]):
            ear = landmarks[indices['ear']]
            shoulder = landmarks[indices['shoulder']]
            hip = landmarks[indices['hip']]
            angles['shoulder'] = angle_between_points_cv2(ear, shoulder, hip)

        # Kalça açısı (omuz-kalça-diz)
        if all(idx < len(landmarks) for idx in [indices['shoulder'], indices['hip'], indices['knee']]):
            shoulder = landmarks[indices['shoulder']]
            hip = landmarks[indices['hip']]
            knee = landmarks[indices['knee']]
            angles['hip'] = angle_between_points_cv2(shoulder, hip, knee)

        # Diz açısı (kalça-diz-ayak bileği)
        if all(idx < len(landmarks) for idx in [indices['hip'], indices['knee'], indices['ankle']]):
            hip = landmarks[indices['hip']]
            knee = landmarks[indices['knee']]
            ankle = landmarks[indices['ankle']]
            angles['knee'] = angle_between_points_cv2(hip, knee, ankle)

        # Ayak bileği açısı (diz-ayak bileği-ayak)
        if all(idx < len(landmarks) for idx in [indices['knee'], indices['ankle']]):
            knee = landmarks[indices['knee']]
            ankle = landmarks[indices['ankle']]
            # Foot index kontrolü
            if indices['foot'] < len(landmarks):
                foot = landmarks[indices['foot']]
                angles['ankle'] = angle_between_points_cv2(knee, ankle, foot)

        angles['selected_side'] = side
        return angles

    except Exception as e:
        print(f"⚠️ Vücut açısı hesaplama hatası: {e}")
        return {}


def calculate_distance(point1, point2):
    """İki nokta arasındaki Euclidean mesafeyi hesapla"""
    try:
        # MediaPipe landmark objesi kontrolü
        if hasattr(point1, 'x') and hasattr(point1, 'y'):
            point1 = (point1.x, point1.y)
        if hasattr(point2, 'x') and hasattr(point2, 'y'):
            point2 = (point2.x, point2.y)

        p1 = np.array(point1, dtype=np.float32)
        p2 = np.array(point2, dtype=np.float32)
        return float(np.linalg.norm(p2 - p1))
    except Exception as e:
        print(f"⚠️ Mesafe hesaplama hatası: {e}")
        return 0.0


def normalize_coordinates(landmarks, frame_shape):
    """
    Landmark koordinatlarını frame boyutuna göre normalize et

    Args:
        landmarks: MediaPipe landmarks
        frame_shape: (height, width, channels)

    Returns:
        list: Normalize edilmiş (x, y) koordinatları
    """
    try:
        height, width = frame_shape[:2]
        normalized = []

        for landmark in landmarks:
            x = landmark.x * width
            y = landmark.y * height
            normalized.append((int(x), int(y)))

        return normalized
    except Exception as e:
        print(f"⚠️ Koordinat normalize hatası: {e}")
        return []


def calculate_pose_stability(landmarks_history, window_size=10):
    """
    Pose kararlılığını hesapla (son N frame'deki değişim)

    Args:
        landmarks_history: Son N frame'deki landmark listesi
        window_size: Analiz edilecek frame sayısı

    Returns:
        float: Kararlılık skoru (0-100, yüksek = daha kararlı)
    """
    try:
        if len(landmarks_history) < 2:
            return 100.0

        # Son window_size kadar frame'i al
        recent_landmarks = landmarks_history[-window_size:]

        # Ana landmark noktalarının varyansını hesapla
        key_points = [11, 12, 23, 24]  # Omuzlar ve kalçalar
        total_variance = 0

        for point_idx in key_points:
            x_coords = [frame[point_idx].x for frame in recent_landmarks
                        if len(frame) > point_idx]
            y_coords = [frame[point_idx].y for frame in recent_landmarks
                        if len(frame) > point_idx]

            if len(x_coords) > 1:
                total_variance += np.var(x_coords) + np.var(y_coords)

        # Variance'ı 0-100 skala skoruna çevir
        stability_score = max(0, 100 - (total_variance * 10000))
        return float(stability_score)

    except Exception as e:
        print(f"⚠️ Kararlılık hesaplama hatası: {e}")
        return 50.0


def draw_angle_arc(image, center, angle, radius=30, color=(255, 255, 0), thickness=2):
    """
    Açıyı görselleştirmek için yay çiz

    Args:
        image: OpenCV image
        center: (x, y) merkez nokta
        angle: Açı değeri (derece)
        radius: Yay yarıçapı
        color: BGR renk
        thickness: Çizgi kalınlığı
    """
    try:
        center = (int(center[0]), int(center[1]))
        cv2.ellipse(image, center, (radius, radius), 0, 0, int(angle), color, thickness)

        # Açı değerini yazı olarak ekle
        text_pos = (center[0] + radius + 5, center[1])
        cv2.putText(image, f"{angle:.1f}°", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except Exception as e:
        print(f"⚠️ Yay çizme hatası: {e}")


def get_landmark_visibility_score(landmarks, indices):
    """
    Belirtilen landmark'ların görünürlük skorunu hesapla

    Args:
        landmarks: MediaPipe landmarks
        indices: Kontrol edilecek landmark indeksleri

    Returns:
        float: Ortalama görünürlük skoru (0-1)
    """
    try:
        if not indices or not landmarks:
            return 0.0

        valid_indices = [i for i in indices if i < len(landmarks)]
        if not valid_indices:
            return 0.0

        total_visibility = sum(landmarks[i].visibility for i in valid_indices)
        return total_visibility / len(valid_indices)
    except Exception as e:
        print(f"⚠️ Görünürlük skoru hatası: {e}")
        return 0.0


def is_pose_valid(landmarks, min_visibility=0.5):
    """
    Pose'un geçerli olup olmadığını kontrol et

    Args:
        landmarks: MediaPipe landmarks
        min_visibility: Minimum görünürlük eşiği

    Returns:
        bool: Pose geçerli mi?
    """
    try:
        if not landmarks:
            return False

        # Ana landmark'ların görünürlüğünü kontrol et
        key_landmarks = [11, 12, 23, 24]  # Omuzlar ve kalçalar
        visibility_score = get_landmark_visibility_score(landmarks, key_landmarks)

        return visibility_score >= min_visibility

    except Exception as e:
        print(f"⚠️ Pose geçerlilik kontrolü hatası: {e}")
        return False


# Test fonksiyonu
if __name__ == "__main__":
    # Test noktaları
    test_points = [
        ((0, 0), (1, 0), (1, 1)),  # 90 derece
        ((0, 0), (1, 0), (2, 0)),  # 180 derece (düz çizgi)
        ((0, 0), (1, 0), (0, 1)),  # 90 derece
        ((0.5, 0.5), (0.6, 0.5), (0.6, 0.6)),  # 90 derece (küçük değerler)
    ]

    print("🧮 Geo Engine Test Sonuçları:")
    for i, (A, B, C) in enumerate(test_points):
        angle_cv2 = angle_between_points_cv2(A, B, C)
        angle_np = angle_between_points_numpy(A, B, C)
        print(f"Test {i + 1}: CV2={angle_cv2:.1f}°, NumPy={angle_np:.1f}°")

    print("✅ Geo Engine hazır ve test edildi!")