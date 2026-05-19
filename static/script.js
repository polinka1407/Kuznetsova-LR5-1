// ============================================
// БЛОК ГЛОБАЛЬНЫХ ПЕРЕМЕННЫХ И НАСТРОЕК
// ============================================
// Назначение: хранение состояния приложения, цветов, порогов уверенности

var bounding_box_colors = {};        // Хранилище цветов для каждого класса объектов
var user_confidence = 0.6;           // Текущий порог уверенности пользователя (60%)

// Палитра цветов для bounding boxes
var color_choices = [
  "#C7FC00", "#FF00FF", "#8622FF", "#FE0056", "#00FFCE",
  "#FF8000", "#00B7EB", "#FFFF00", "#0E7AFE", "#FFABAB",
  "#0000FF", "#CCCCCC",
];

var canvas_painted = false;           // Флаг инициализации canvas
var canvas = document.getElementById("video_canvas");  // Получение canvas
var ctx = canvas.getContext("2d");    // Контекст для рисования

const inferEngine = new inferencejs.InferenceEngine();  // Движок инференса Roboflow
var modelWorkerId = null;             // ID worker модели


// ============================================
// БЛОК ЗАХВАТА ВИДЕО И ИНИЦИАЛИЗАЦИИ МОДЕЛИ
// ============================================
// Назначение: запрос доступа к веб-камере, настройка видео, загрузка модели

function webcamInference() {
  // Отображение индикатора загрузки
  var loading = document.getElementById("loading");
  loading.style.display = "block";

  // ЗАПРОС ДОСТУПА К ВЕБ-КАМЕРЕ
  navigator.mediaDevices
    .getUserMedia({ video: { facingMode: "environment" } })  // тыловая камера
    .then(function(stream) {
      // СОЗДАНИЕ ВИДЕОЭЛЕМЕНТА
      video = document.createElement("video");
      video.srcObject = stream;        // Привязка потока с камеры
      video.id = "video1";
      video.style.display = "none";    // Скрыто до готовности
      video.setAttribute("playsinline", "");  // Для iOS

      document.getElementById("video_canvas").after(video);

      // НАСТРОЙКА РАЗМЕРОВ ПРИ ЗАГРУЗКЕ ВИДЕО
      video.onplay = function() {
        height = video.videoHeight;
        width = video.videoWidth;

        // Установка размеров видео (640x480)
        video.width = width;
        video.height = height;
        video.style.width = 640 + "px";
        video.style.height = 480 + "px";

        // Установка размеров canvas
        canvas.style.width = 640 + "px";
        canvas.style.height = 480 + "px";
        canvas.width = width;
        canvas.height = height;

        document.getElementById("video_canvas").style.display = "block";
      };

      // ЗАГРУЗКА И ЗАПУСК МОДЕЛИ ROBOFLOW
      inferEngine.startWorker(
        MODEL_NAME,          // "microsoft-coco" из index.html
        MODEL_VERSION,       // 9 из index.html
        publishable_key,     // API ключ из index.html
        [{ scoreThreshold: CONFIDENCE_THRESHOLD }]  // порог из index.html
      ).then((id) => {
        modelWorkerId = id;  // Сохранение ID worker'а
        detectFrame();       // ЗАПУСК ОСНОВНОГО ЦИКЛА
      });
    })
    .catch(function(err) {
      console.log(err);      // Обработка ошибок (нет камеры, отказано в доступе)
    });
}


// ============================================
// БЛОК ОСНОВНОГО ЦИКЛА ИНФЕРЕНСА
// ============================================
// Назначение: рекурсивный захват кадров, отправка в модель, отрисовка

function detectFrame() {
  // Проверка готовности модели
  if (!modelWorkerId) return requestAnimationFrame(detectFrame);

  // ВЫПОЛНЕНИЕ ИНФЕРЕНСА НА ТЕКУЩЕМ КАДРЕ
  inferEngine.infer(modelWorkerId, new inferencejs.CVImage(video))
    .then(function(predictions) {

      // ИНИЦИАЛИЗАЦИЯ CANDPAS ПРИ ПЕРВОМ ЗАПУСКЕ
      if (!canvas_painted) {
        var video_start = document.getElementById("video1");
        
        // Позиционирование canvas поверх видео
        canvas.top = video_start.top;
        canvas.left = video_start.left;
        canvas.style.top = video_start.top + "px";
        canvas.style.left = video_start.left + "px";
        canvas.style.position = "absolute";
        video_start.style.display = "block";
        canvas.style.display = "absolute";
        canvas_painted = true;

        // Скрытие индикатора загрузки
        var loading = document.getElementById("loading");
        loading.style.display = "none";
      }

      // РЕКУРСИВНЫЙ ВЫЗОВ ДЛЯ СЛЕДУЮЩЕГО КАДРА
      requestAnimationFrame(detectFrame);
      
      // ОЧИСТКА И ПЕРЕРИСОВКА
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (video) {
        drawBoundingBoxes(predictions, ctx)  // ВЫЗОВ ОТРИСОВКИ
      }
    });
}


// ============================================
// БЛОК ОБРАБОТКИ И ВИЗУАЛИЗАЦИИ ПРЕДСКАЗАНИЙ
// ============================================
// Назначение: фильтрация предсказаний, отрисовка bounding boxes и текста

function drawBoundingBoxes(predictions, ctx) {
  // ПЕРЕБОР ВСЕХ РАСПОЗНАННЫХ ОБЪЕКТОВ
  for (var i = 0; i < predictions.length; i++) {
    var confidence = predictions[i].confidence;

    // ФИЛЬТРАЦИЯ ПО ПОРОГУ УВЕРЕННОСТИ
    if (confidence < user_confidence) {
      continue  // Пропуск объектов с низкой уверенностью
    }

    // НАЗНАЧЕНИЕ ЦВЕТА ДЛЯ КЛАССА ОБЪЕКТА
    if (predictions[i].class in bounding_box_colors) {
      // Использование существующего цвета
      ctx.strokeStyle = bounding_box_colors[predictions[i].class];
    } else {
      // Выбор случайного цвета из палитры
      var color = color_choices[Math.floor(Math.random() * color_choices.length)];
      ctx.strokeStyle = color;
      // Удаление использованного цвета из палитры
      color_choices.splice(color_choices.indexOf(color), 1);
      // Сохранение цвета для класса
      bounding_box_colors[predictions[i].class] = color;
    }

    // РАСЧЕТ КООРДИНАТ BOUNDING BOX
    var prediction = predictions[i];
    var x = prediction.bbox.x - prediction.bbox.width / 2;   // Левый верхний X
    var y = prediction.bbox.y - prediction.bbox.height / 2;  // Левый верхний Y
    var width = prediction.bbox.width;
    var height = prediction.bbox.height;

    // ОТРИСОВКА ПРЯМОУГОЛЬНИКА
    ctx.rect(x, y, width, height);
    ctx.fillStyle = "rgba(0, 0, 0, 0)";  // Прозрачная заливка
    ctx.fill();
    ctx.lineWidth = "4";                  // Толщина линии
    ctx.strokeRect(x, y, width, height);  // Рисование рамки

    // ОТРИСОВКА ТЕКСТА (класс + процент уверенности)
    ctx.font = "25px Arial";
    ctx.fillStyle = ctx.strokeStyle;      // Текст цветом рамки
    ctx.fillText(
      prediction.class + " " + Math.round(confidence * 100) + "%",
      x, y - 10                           // Позиция над рамкой
    );
  }
}


// ============================================
// БЛОК УПРАВЛЕНИЯ ПОРОГОМ УВЕРЕННОСТИ
// ============================================
// Назначение: обработка ввода пользователя, обновление фильтрации

function changeConfidence() {
  // СЧИТЫВАНИЕ ЗНАЧЕНИЯ С ПОЛЗУНКА (1-100) И ПРЕОБРАЗОВАНИЕ В ДРОБЬ (0.01-1.00)
  user_confidence = document.getElementById("confidence").value / 100;
}

// НАСТРОЙКА ОБРАБОТЧИКА СОБЫТИЙ
document.getElementById("confidence").addEventListener("input", changeConfidence);


// ============================================
// БЛОК ЗАПУСКА ПРИЛОЖЕНИЯ
// ============================================
// Назначение: инициализация приложения при загрузке страницы

webcamInference();
