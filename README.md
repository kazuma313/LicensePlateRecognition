# LicensePlateRecognition

- Deteksi plate menggunakan [huggingface]{https://huggingface.co/skiba4/license_plate} model dan rekognasi tulisan menggunakan CNN, RNN, dan implementasi CTC loss.
- rekognasi tulisan menggunakan easyOCR.

## Dataset

dataset yang digunakan berjumlah **371** dataset pada folder _.\dataset\platGray_

## Usage

pastikan libray telah di install di **requirements.txt**.

### Menjalankan model yang menggunakan CNN, RNN, dan implementasi CTC loss menggunakan 371 dataset:

```
python camera.py
```

hasil ScreenShoot:
![Screen shoot menggunakan huggingface model dan model CNN, RNN, dan implementasi CTC loss untuk rekognasi text pada kamera](https://github.com/kazuma313/LicensePlateRecognition/plat_detected.jpg)

### menjalan model yang menggunakan easyOCR:

```
python camera_easyOCR.py
```

hasil ScreenShoot:
![Screen shoot menggunakan model easyOCR untuk rekognasi text pada kamera](https://github.com/kazuma313/LicensePlateRecognition/open_Cv_easyOCR0.jpg)

## Deskiripsi

Saya membandingkan 2 model untuk membandingkan dan mendapatkan kesimpulan:

- Menggunakan CNN, RNN, dan CTC Loss.
- easy OCR

### Menggunakan CNN, RNN, dan CTC Loss.

Karna keterbatas dataset, model gagal memprediksi karakter. Jika memiliki data yang banyak, maka model akan mengenali karakter dan dapat mengantisipasi noise pada gambar.

Untuk mendeteksi apakah terdapat image plate atau tidak, saya menggunakan model yang tersedia dari [huggingface]{https://huggingface.co/skiba4/license_plate}. Sehingga, jika frame pada vidio terdeteksi pelat atau tidak, maka akan dilakukan rekognasi text.

### Menggunakan easyOCR

Dapat mengenali krakter dengan baik. Jika disediakan gambar yang memiliki noise, maka easyOCR akan kesulitan mengenali karakter.
