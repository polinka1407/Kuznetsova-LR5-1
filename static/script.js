// ============================================
// БЛОК 1: ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И НАСТРОЙКИ
// ============================================
// Назначение: хранение состояния приложения, цветов для bounding boxes, порогов уверенности

// Хранилище цветов для каждого уникального класса объектов (например, "person" -> "#FF0000")
var bounding_box_colors = {};

// Текущий порог уверенности пользователя (по умолчанию 60% = 0.6)
var user_confidence = 0.6;

// Палитра цветов для bounding boxes (12 предопределенных цветов)
var color_choices = [
  "#C7FC00", "#FF00FF", "#8622FF", "#FE0056", "#00FFCE",
  "#FF8000", "#00B7EB", "#FFFF00", "#0E7AFE", "#FFABAB",
  "#0000FF", "#CCCCCC"
];

// Флаг: был ли уже инициализирован canvas (первый кадр)
var canvas_painted = false;

// Получение ссылки на canvas элемент из DOM
var canvas = document.getElementById("video_canvas");
// Получение 2D контекста для рисования на canvas
var ctx = canvas.getContext("2d");

// Создание движка инференса Roboflow
const inferEngine = new inferencejs.InferenceEngine();
// ID worker модели (будет заполнен после загрузки модели)
var modelWorkerId = null;


// ============================================
// БЛОК 2: ЗАХВАТ ВИДЕО С ВЕБ-КАМЕРЫ И ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
// ============================================
// Назначение: запрос доступа к веб-камере, настройка видеоэлемента, загрузка модели Roboflow

function webcamInference() {
  // БЛОК 2.1: ОТОБРАЖЕНИЕ ИНДИКАТОРА ЗАГРУЗКИ
  var loading = document.getElementById("loading");
  loading.style.display = "block";

  // БЛОК 2.2: ЗАПРОС ДОСТУПА К ВЕБ-КАМЕРЕ
  // facingMode: "environment" - использует тыловую камеру на мобильных устройствах
  navigator.mediaDevices
    .getUserMedia({ video: { facingMode: "environment" } })
    .then(function(stream) {
      // БЛОК 2.3: СОЗДАНИЕ ВИДЕОЭЛЕМЕНТА И ПРИВЯЗКА ПОТОКА
      video = document.createElement("video");
      video.srcObject = stream;      // Привязка видеопотока с камеры
      video.id = "video1";
      video.style.display = "none";   // Скрываем до полной готовности
      video.setAttribute("playsinline", "");  // Необходимо для iOS

      // Вставка видео после canvas в DOM
      document.getElementById("video_canvas").after(video);

      // БЛОК 2.4: ОБРАБОТЧИК ЗАГРУЗКИ МЕТАДАННЫХ ВИДЕО
      video.onloadedmetadata = function() {
        video.play();  // Запуск воспроизведения
      }

      // БЛОК 2.5: НАСТРОЙКА РАЗМЕРОВ ВИДЕО И CANVAS (когда видео готово)
      video.onplay = function() {
        height = video.videoHeight;
        width = video.videoWidth;

        // Установка отображаемых размеров видео (640x480)
        video.width = width;
        video.height = height;
        video.style.width = 640 + "px";
        video.style.height = 480 + "px";

        // Установка размеров canvas (соответствует видео)
        canvas.style.width = 640 + "px";
        canvas.style.height = 480 + "px";
        canvas.width = width;
        canvas.height = height;

        // Показываем canvas
        document.getElementById("video_canvas").style.display = "block";
      };

      // БЛОК 2.6: МАСШТАБИРОВАНИЕ КОНТЕКСТА РИСОВАНИЯ
      ctx.scale(1, 1);

      // БЛОК 2.7: ЗАГРУЗКА И ЗАПУСК МОДЕЛИ ROBOFLOW
      // startWorker загружает модель и создает worker для асинхронной обработки
      inferEngine.startWorker(
        MODEL_NAME,                           // "microsoft-coco"
        MODEL_VERSION,                        // 9
        publishable_key,                      // API ключ
        [{ scoreThreshold: CONFIDENCE_THRESHOLD }]  // Минимальный порог уверенности
      )
        .then((id) => {
          modelWorkerId = id;                 // Сохраняем ID worker'а
          detectFrame();                      // ЗАПУСК ОСНОВНОГО ЦИКЛА ОБРАБОТКИ
        });
    })
    // БЛОК 2.8: ОБРАБОТКА ОШИБОК ДОСТУПА К КАМЕРЕ
    .catch(function(err) {
      console.log(err);  // Вывод ошибки в консоль (нет камеры или отказано в доступе)
    });
}


// ============================================
// БЛОК 3: ОСНОВНОЙ ЦИКЛ ИНФЕРЕНСА (ОБРАБОТКА КАЖДОГО КАДРА)
// ============================================
// Назначение: рекурсивный захват кадров, отправка в модель, получение предсказаний

function detectFrame() {
  // БЛОК 3.1: ПРОВЕРКА ГОТОВНОСТИ МОДЕЛИ
  // Если модель еще не загружена, просто запрашиваем следующий кадр
  if (!modelWorkerId) return requestAnimationFrame(detectFrame);

  // БЛОК 3.2: ВЫПОЛНЕНИЕ ИНФЕРЕНСА НА ТЕКУЩЕМ КАДРЕ
  // CVImage преобразует видео/изображение в формат, понятный модели
  inferEngine.infer(modelWorkerId, new inferencejs.CVImage(video))
    .then(function(predictions) {
      
      // БЛОК 3.3: ИНИЦИАЛИЗАЦИЯ CANVAS ПРИ ПЕРВОМ ЗАПУСКЕ
      if (!canvas_painted) {
        var video_start = document.getElementById("video1");

        // Позиционирование canvas поверх видео (абсолютное позиционирование)
        canvas.top = video_start.top;
        canvas.left = video_start.left;
        canvas.style.top = video_start.top + "px";
        canvas.style.left = video_start.left + "px";
        canvas.style.position = "absolute";
        video_start.style.display = "block";
        canvas.style.display = "absolute";
        canvas_painted = true;  // Флаг установлен - повторная инициализация не нужна

        // Скрываем индикатор загрузки
        var loading = document.getElementById("loading");
        loading.style.display = "none";
      }
      
      // БЛОК 3.4: РЕКУРСИВНЫЙ ВЫЗОВ ДЛЯ СЛЕДУЮЩЕГО КАДРА
      // requestAnimationFrame синхронизируется с частотой обновления экрана (обычно 60 FPS)
      requestAnimationFrame(detectFrame);
      
      // БЛОК 3.5: ОЧИСТКА И ПЕРЕРИСОВКА КАДРА
      ctx.clearRect(0, 0, canvas.width, canvas.height);  // Очищаем canvas
      if (video) {
        drawBoundingBoxes(predictions, ctx);  // Отрисовка bounding boxes
      }
    });
}


// ============================================
// БЛОК 4: ОБРАБОТКА ПРЕДСКАЗАНИЙ И ВИЗУАЛИЗАЦИЯ (ОТРИСОВКА BOUNDING BOXES)
// ============================================
// Назначение: фильтрация предсказаний по уверенности, отрисовка рамок и текста на canvas

function drawBoundingBoxes(predictions, ctx) {
  // БЛОК 4.1: ПЕРЕБОР ВСЕХ РАСПОЗНАННЫХ ОБЪЕКТОВ
  for (var i = 0; i < predictions.length; i++) {
    var confidence = predictions[i].confidence;

    // БЛОК 4.2: ФИЛЬТРАЦИЯ ПО ПОРОГУ УВЕРЕННОСТИ
    // Пропускаем объекты с низкой уверенностью (ниже значения ползунка)
    if (confidence < user_confidence) {
      continue;  // Переход к следующему предсказанию
    }

    // БЛОК 4.3: НАЗНАЧЕНИЕ ЦВЕТА ДЛЯ КЛАССА ОБЪЕКТА
    // Каждый уникальный класс объектов получает свой цвет
    if (predictions[i].class in bounding_box_colors) {
      // Если цвет уже был назначен этому классу - используем его
      ctx.strokeStyle = bounding_box_colors[predictions[i].class];
    } else {
      // Иначе выбираем случайный цвет из палитры
      var color = color_choices[Math.floor(Math.random() * color_choices.length)];
      ctx.strokeStyle = color;
      // Удаляем использованный цвет из палитры (чтобы не повторялся)
      color_choices.splice(color_choices.indexOf(color), 1);
      // Сохраняем цвет для этого класса объектов
      bounding_box_colors[predictions[i].class] = color;
    }

    // БЛОК 4.4: РАСЧЕТ КООРДИНАТ BOUNDING BOX
    // Модель возвращает центр бокса (x, y) и размеры (width, height)
    // Нужно пересчитать в левый верхний угол для отрисовки
    var prediction = predictions[i];
    var x = prediction.bbox.x - prediction.bbox.width / 2;   // X левого верхнего угла
    var y = prediction.bbox.y - prediction.bbox.height / 2;  // Y левого верхнего угла
    var width = prediction.bbox.width;
    var height = prediction.bbox.height;

    // БЛОК 4.5: ОТРИСОВКА ПРЯМОУГОЛЬНИКА (BOUNDING BOX)
    ctx.rect(x, y, width, height);      // Создание пути прямоугольника
    ctx.fillStyle = "rgba(0, 0, 0, 0)"; // Прозрачная заливка (только рамка)
    ctx.fill();                         // Применение заливки
    ctx.lineWidth = "4";                // Толщина линии рамки (4 пикселя)
    ctx.strokeRect(x, y, width, height); // Рисование рамки

    // БЛОК 4.6: ОТРИСОВКА ТЕКСТА (название класса + процент уверенности)
    ctx.font = "25px Arial";            // Шрифт текста
    ctx.fillStyle = ctx.strokeStyle;    // Цвет текста совпадает с цветом рамки
    ctx.fillText(
      prediction.class + " " + Math.round(confidence * 100) + "%",  // Текст: "person 85%"
      x,                                // Позиция X (левый край рамки)
      y - 10                            // Позиция Y (НАД рамкой, отступ 10px)
    );
  }
}


// ============================================
// БЛОК 5: УПРАВЛЕНИЕ ПОРОГОМ УВЕРЕННОСТИ (ОБРАБОТЧИК ПОЛЗУНКА)
// ============================================
// Назначение: чтение значения ползунка и обновление порога фильтрации

function changeConfidence() {
  // Получаем значение ползунка (от 1 до 100) и делим на 100
  // Преобразование: 60 -> 0.6, 75 -> 0.75 и т.д.
  user_confidence = document.getElementById("confidence").value / 100;
}

// Регистрация обработчика события "input" для ползунка
// Функция changeConfidence будет вызываться при каждом движении ползунка
document.getElementById("confidence").addEventListener("input", changeConfidence);


// ============================================
// БЛОК 6: ЗАПУСК ПРИЛОЖЕНИЯ
// ============================================
// Назначение: инициализация всего приложения при загрузке страницы
webcamInference();
