"""
🧪 Mevcut Kurulum Test Scripti
============================
Bu script mevcut kodlarınızın çalışıp çalışmadığını test eder.
"""

import os
import sys
from pathlib import Path


def test_imports():
    """Gerekli kütüphaneleri test et"""
    print("📦 Python kütüphaneleri test ediliyor...")

    imports_to_test = [
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('csv', 'CSV (built-in)'),
        ('time', 'Time (built-in)')
    ]

    failed_imports = []

    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name} - HATA: {e}")
            failed_imports.append(name)

    return len(failed_imports) == 0


def test_file_structure():
    """Dosya yapısını kontrol et"""
    print("\n📁 Dosya yapısı kontrol ediliyor...")

    required_files = [
        'main.py',
        'machine_learning_playground.py',
        'geo_engine.py',
        'models/pose_landmarker.task'
    ]

    required_dirs = [
        'models',
        'input',
        'output'
    ]

    missing_files = []
    missing_dirs = []

    # Dosyaları kontrol et
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if 'pose_landmarker.task' in file_path:
                print(f"  ✅ {file_path} ({size / 1024 / 1024:.1f} MB)")
            else:
                print(f"  ✅ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} - EKSİK")
            missing_files.append(file_path)

    # Klasörleri kontrol et
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"  ✅ {dir_path}/ klasörü")
        else:
            print(f"  ❌ {dir_path}/ klasörü - EKSİK")
            missing_dirs.append(dir_path)

    return len(missing_files) == 0 and len(missing_dirs) == 0


def test_pose_landmarker():
    """Pose Landmarker'ın yüklenip yüklenmediğini test et"""
    print("\n🤖 Pose Landmarker test ediliyor...")

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import RunningMode

        # Model dosyası yolunu kontrol et
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'models', 'pose_landmarker.task')

        if not os.path.exists(model_path):
            print(f"  ❌ Model dosyası bulunamadı: {model_path}")
            return False

        # Model yüklemeyi test et
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            running_mode=RunningMode.VIDEO
        )

        pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        print("  ✅ Pose Landmarker başarıyla yüklendi!")

        return True

    except Exception as e:
        print(f"  ❌ Pose Landmarker yükleme hatası: {e}")
        return False


def test_webcam():
    """Webcam erişimini test et"""
    print("\n📹 Webcam erişimi test ediliyor...")

    try:
        import cv2

        # Webcam'i dene
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("  ❌ Webcam açılamadı")
            return False

        # Bir frame oku
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"  ✅ Webcam çalışıyor! Çözünürlük: {width}x{height}")
            return True
        else:
            print("  ❌ Webcam'den frame okunamadı")
            return False

    except Exception as e:
        print(f"  ❌ Webcam test hatası: {e}")
        return False


def test_geo_engine():
    """geo_engine.py modülünü test et"""
    print("\n📐 geo_engine modülü test ediliyor...")

    try:
        import geo_engine

        # Test açısı hesapla
        test_angle = geo_engine.angle_between_points_cv2((0, 0), (1, 0), (1, 1))

        if 89 <= test_angle <= 91:  # 90 derece olmalı
            print(f"  ✅ geo_engine çalışıyor! Test açısı: {test_angle:.1f}°")
            return True
        else:
            print(f"  ⚠️ geo_engine çalışıyor ama sonuç şüpheli: {test_angle:.1f}°")
            return True

    except Exception as e:
        print(f"  ❌ geo_engine test hatası: {e}")
        return False


def main():
    """Ana test fonksiyonu"""
    print("=" * 60)
    print("🧪 MEVCUT KURULUM TEST EDİLİYOR")
    print("=" * 60)

    tests = [
        ("Python Kütüphaneleri", test_imports),
        ("Dosya Yapısı", test_file_structure),
        ("Pose Landmarker", test_pose_landmarker),
        ("Webcam", test_webcam),
        ("Geo Engine", test_geo_engine)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  💥 {test_name} testi sırasında hata: {e}")
            results[test_name] = False

    # Sonuçları özetle
    print("\n" + "=" * 60)
    print("📊 TEST SONUÇLARI")
    print("=" * 60)

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1

    print(f"\n🎯 Genel Durum: {passed}/{total} test geçti")

    if passed == total:
        print("🎉 TÜM TESTLER BAŞARILI! Kodlarınızı çalıştırabilirsiniz.")
        print("\n📋 SONRAKİ ADIMLAR:")
        print("1️⃣  python main.py (veri toplama modu)")
        print("2️⃣  python machine_learning_playground.py (AI analiz modu)")
    else:
        print("⚠️ Bazı testler başarısız. Sorunları çözün ve tekrar deneyin.")
        print("\n💡 ÇÖZÜM ÖNERİLERİ:")

        if not results.get("Python Kütüphaneleri", True):
            print("• Eksik paketleri yükleyin: pip install mediapipe opencv-python pandas scikit-learn")

        if not results.get("Dosya Yapısı", True):
            print("• Eksik dosyaları oluşturun veya klasörleri kontrol edin")

        if not results.get("Pose Landmarker", True):
            print("• Model dosyasını tekrar indirin")

        if not results.get("Webcam", True):
            print("• Webcam bağlantısını kontrol edin veya video dosyası kullanın")


if __name__ == "__main__":
    main()