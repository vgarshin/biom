# biom
Entrance face recognition tools

Работает следующим образом. Сначала запускаем процесс записи изображений с камеры с определенным интервалом:
python startrec.py ip destination_folder delay total_images
где:
ip - IP адрес и порт камеры
destination_folder - папка, куда записываются изображения (кадры)
delay - задержка между кадрами
total_images - общее количество кадров, которое необходимо записать с камеры

Пример:
python startrec.py rtsp://admin:SWSCFX@192.168.10.159 shots 1 10
