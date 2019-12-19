# biom
## Entrance face recognition tools

Работает следующим образом. Сначала запускаем процесс записи изображений с камеры с определенным интервалом:
```
python startrec_mt.py destination_folder delay total_images
```
где:
- ip - IP адрес и порт камеры
- destination_folder - папка, куда записываются изображения (кадры)
- delay - задержка между кадрами в сек.
- total_images - общее количество кадров, которое необходимо записать с камеры

Пример:
```
python startrec.py shots .5 1000
```
Следующим шагом запускаем процесс распознавания изображений:

где:
- data_base_create - флаг создания базы данных 
- photos_folder - путь, где лежат фото людей, которых мы распознаем
- shots_folder - путь, где лежат кадры с камеры
- shots_processed_folder - путь, куда перемещаются обработанные кадры
- logs_folder - путь для сохранения логов
- starttime - пока не используется
- level - уровень принятия решений (порог отсечения)

Пример:
```
python startrec.py nodbcreate photos shots shotsprcd logs starttime .7
```
