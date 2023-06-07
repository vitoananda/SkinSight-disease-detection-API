# How to use
clone repository dengan 
```
  git clone https://github.com/vitoananda/SkinSight-disease-detection-API.git
```

Kemudian masukan file keyfile.json yang anda dapatkan dari service account anda yang memiliki role Cloud Storage Admin, Cloud Storage Creator, dan Cloud Storage Viewer

Kemudian masukan juga file serviceAccount.json yang didapatkan dari console project firebase anda

Kemudian ubah isi file .env anda dengan konfigurasi project firebase dan google cloud console anda

# Endpoints

| Method | Endpoint           |
| ------ | ------------------ |  
| POST   | /detect-disease/{uid}           | 
| GET    | /history/{uid}           | 



<hr>

### <b>POST /detect-disease/{uid}</b>
Melakukan upload foto kemudian mendapatkan result prediction yang dihasilkan setelah image diproses. 

Request parameter:
uid: uid user

Request body: 
<p align="left"> <img src="./documentation asset/disease body.jpg" width="700" height="300" /> </p>
Response: 

```json
{
    "type" : "Skin Disease Detection",
    "status": "Success",
    "message": "Deteksi penyakit berhasil",
    "detection_img": "{public_url}",
    "class" : "{predicted_class}"
}
```
200 Jika deteksi penyakit berhasil

```json
{
  "status": "Failed",
  "message": "Tidak ada file yang ditambahkan"
}
```
400 Jika tidak ada file yang ditambahkan


<hr>

### <b>GET /history/{uid}</b>
Mengambil informasi history user berdasarkan uid. 

Request parameter:
uid: uid user

Response:

```json
{
  "status": "Success",
  "message": "Skin detection history berhasil didapatkan",
  "data": {
        "datetime": "Wed, 07 Jun 2023 06:51:39 GMT",
        "predicted_class": "Actinic Keratosis",
        "detection_img": "{public_url}",
        "type": "Skin Disease Detection"
    }
}
```
200 Jika skin detection history berhasil didapatkan

```json
{
   "status": "Failed",
   "message": "User tidak ditemukan"
}
```
404 Jika user tidak ditemukan

