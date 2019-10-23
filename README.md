# biom
## Entrance face recognition tools

Работает следующим образом. Сначала запускаем процесс записи изображений с камеры с определенным интервалом:

python startrec.py ip destination_folder delay total_images

где:

ip - IP адрес и порт камеры
destination_folder - папка, куда записываются изображения (кадры)
delay - задержка между кадрами
total_images - общее количество кадров, которое необходимо записать с камеры

Пример:

python startrec.py rtsp://admin:SWSCFX@192.168.10.159 shots 1 10

Следующим шагом запускаем процесс разспознавания изображений:

где:

data_base_create - флаг создания базы данных 
photos_folder - путь, где лежат фото людей, которых мы распознаем
shots_folder - путь, где лежат кадры с камеры
shots_processed_folder - путь, куда перемещаются обраотанные кадры
logs_folder - путь для сохранения логов
starttime - пока не используется

Пример:

python startrec.py dbcreate photos shots shotsprcd logs starttime
