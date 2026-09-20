"use strict";
const $ = (id) => document.getElementById(id);
const emptyAlerts = $("alerts").firstElementChild.cloneNode(true);
const fallbackLabels = {missing_api_key:"не указан ключ API", invalid_api_key:"неверный ключ API", access_denied:"нет доступа к модели", model_not_available:"модель недоступна", provider_rate_limit:"лимит Groq", provider_timeout:"Groq не ответил вовремя", provider_unreachable:"нет соединения с провайдером", invalid_provider_response:"некорректный ответ модели", rate_limited_locally:"интервал между VLM-запросами — 20 секунд", missing_incident_image:"нет снимка инцидента"};
let running = false, busy = false, demoReady = false, lastAlerts = "", frameBusy = false;
let latestStatus = null;
const notes = {
  demo: "Запись OpenCV с настоящими пешеходами. Детектор YOLOv8n работает локально на CPU.",
  webcam: "Используется камера компьютера, на котором запущен сервер. При наличии скачанной модели выбирается YOLO; иначе — CPU HOG.",
  custom: "Путь должен быть доступен серверу. RTSP — прямое подключение к вашей камере; видеофайл воспроизводится по кругу.",
  synthetic: "Тестовый прямоугольник вместо человека. Это симуляция, а не реальные данные."
};
function message(text, error = false) { $("message").textContent = text; $("message").hidden = !text; $("message").classList.toggle("error", error); }
async function api(path, options) {
  const response = await fetch(path, {cache: "no-store", ...options});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Проверьте параметры: " + JSON.stringify(data.detail));
  return data;
}
function controls() { $("start").disabled = busy || running; $("stop").disabled = busy || !running; $("source").disabled = busy || running; }
$("source").addEventListener("change", () => {
  $("customLabel").hidden = $("source").value !== "custom";
  $("customSource").required = $("source").value === "custom";
  $("sourceNote").textContent = notes[$("source").value];
  message("");
});
$("controls").addEventListener("submit", async (event) => {
  event.preventDefault();
  if ($("source").value === "demo" && !demoReady) {
    message("Подготовьте реальное демо в терминале: python -m pip install -r requirements-onnx.txt, затем python -m src.assets. После этого обновите страницу.", true); return;
  }
  const choice = $("source").value;
  const source = choice === "webcam" ? 0 : choice === "custom" ? $("customSource").value.trim() : choice;
  const zone = ["left", "top", "right", "bottom"].map(id => Number($(id).value));
  if (zone[0] >= zone[2] || zone[1] >= zone[3]) { message("Левая граница должна быть меньше правой, а верхняя — меньше нижней.", true); return; }
  busy = true; controls(); message("Открываем видео и загружаем детектор…");
  try {
    const status = await api("/start-stream", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({source, zone, confidence: Number($("confidence").value), dwell_seconds: Number($("dwell").value), cooldown_seconds: Number($("cooldown").value)})});
    $("video").hidden = true; $("placeholder").hidden = false;
    applyStatus(status);
    message(status.backend === "opencv_hog" ? "Используется запасной CPU HOG-детектор. Для YOLO установите ONNX Runtime и скачайте модель." : "");
  } catch(error) { message(error.message, true); }
  finally { busy = false; controls(); }
});
$("stop").addEventListener("click", async () => {
  busy = true; controls();
  try { applyStatus(await api("/stop-stream", {method:"POST"})); message("Поток остановлен. События и последний кадр доступны до перезапуска сервера."); }
  catch(error) { message(error.message, true); }
  finally { busy = false; controls(); }
});
function applyStatus(status) {
  latestStatus = status; running = status.running;
  $("people").textContent = status.people;
  $("inZone").textContent = status.in_zone;
  $("fps").textContent = status.processing_fps;
  $("latency").textContent = status.latency_ms;
  $("dropped").textContent = status.dropped_frames;
  $("backend").textContent = status.backend || "ожидает запуска";
  $("streamState").textContent = status.error ? "Ошибка источника" : running ? "Обработка идет" : "Остановлен";
  $("streamState").classList.toggle("active", running);
  $("statusDot").classList.toggle("active", running);
  $("stale").hidden = $("video").hidden || (running && (status.frame_age_seconds === null || status.frame_age_seconds < 3));
  $("stale").textContent = running ? "Нет свежих кадров · проверьте источник" : "Поток остановлен · последний кадр";
  const labels = {recording:"ЗАПИСЬ · РЕАЛЬНЫЕ КАДРЫ", simulation:"СИМУЛЯЦИЯ · ТЕСТОВЫЕ КАДРЫ", live:"КАМЕРА · ПРЯМОЙ ПОТОК"};
  $("videoLabel").textContent = labels[status.source_kind] || "";
  $("videoLabel").hidden = !status.source_kind;
  if (status.report_mode === "vlm") {
    $("reportMode").textContent = `Groq Vision · ${status.report_model}. Снимок инцидента передается в Groq, не чаще раза в 20 секунд. ` + (!status.report_key_configured ? "Ключ API не настроен." : status.report_last_error ? "Последняя ошибка: " + (fallbackLabels[status.report_last_error] || status.report_last_error) : "Визуальная оценка + отчет на русском.");
  } else {
    $("reportMode").textContent = status.report_mode === "llm" ? "Отчеты через внешний LLM. При ошибке — локальный шаблон; источник указан у события." : "Правила принятия решений + локальный отчет. Внешний LLM не подключен.";
  }
  if (status.error) message(status.error, true);
  controls();
}
function renderAlerts(alerts) {
  const signature = alerts.map(a => a.incident.id).join(",");
  $("alertCount").textContent = alerts.length;
  if (signature === lastAlerts) return;
  lastAlerts = signature;
  if (!alerts.length) { $("alerts").replaceChildren(emptyAlerts.cloneNode(true)); return; }
  const expanded = new Set(Array.from($("alerts").querySelectorAll(".alert")).filter(node => node.querySelector("details").open).map(node => node.dataset.incident));
  const scrollTop = $("alerts").scrollTop;
  $("alerts").replaceChildren();
  for (const alert of alerts) {
    const node = $("alertTemplate").content.cloneNode(true), event = alert.incident;
    node.querySelector(".alert").dataset.incident = event.id;
    node.querySelector("details").open = expanded.has(event.id);
    const severity = node.querySelector(".severity");
    severity.textContent = alert.severity === "critical" ? "СРОЧНАЯ ПРОВЕРКА" : "ВНИМАНИЕ";
    severity.classList.toggle("critical", alert.severity === "critical");
    node.querySelector("time").textContent = new Date(event.timestamp).toLocaleTimeString("ru-RU");
    node.querySelector("h3").textContent = `Людей в запретной зоне: ${event.detections.length}`;
    node.querySelector(".alert-action").textContent = alert.action === "request_urgent_review" ? "Нужна срочная проверка оператором." : "Проверьте присутствие человека в зоне.";
    const kinds = {simulation:"Симуляция", recording:"Видеозапись", live:"Камера"};
    const sources = {mock:"Локальный шаблон", mock_fallback:"Локальный шаблон", llm:"Отчет LLM", vlm:"Groq · визуальная оценка"};
    node.querySelector(".alert-meta").textContent = `${kinds[event.source_kind] || "Источник неизвестен"} · Кадр ${event.frame_id} · ${event.backend} · ${sources[alert.report_source]}`;
    node.querySelector(".report").textContent = alert.report;
    if (alert.fallback_reason) {
      node.querySelector(".alert-meta").textContent += " · " + (fallbackLabels[alert.fallback_reason] || alert.fallback_reason);
    }
    if (alert.vision_assessment) {
      const assessment = document.createElement("p");
      assessment.className = "vision-assessment";
      const verdicts = {confirmed:"VLM: присутствие подтверждено", not_confirmed:"VLM: присутствие не подтверждено", uncertain:"VLM: недостаточно уверенности"};
      assessment.textContent = verdicts[alert.vision_assessment.verdict] + ". " + alert.vision_assessment.explanation;
      node.querySelector(".alert-action").after(assessment);
    }
    // Construct a local snapshot URL, never follow arbitrary URLs from received webhooks.
    const link = node.querySelector(".snapshot-link");
    if (event.snapshot_url) {
      link.href = `/alerts/${encodeURIComponent(event.id)}/snapshot.jpg`;
      const image = node.querySelector(".snapshot"); image.src = link.href;
      image.addEventListener("error", () => { link.hidden = true; });
    } else link.hidden = true;
    $("alerts").append(node);
  }
  $("alerts").scrollTop = scrollTop;
}
async function poll() {
  try {
    const [status, alerts] = await Promise.all([api("/health"), api("/get-latest-alerts?limit=200")]);
    applyStatus(status); renderAlerts(alerts); $("connection").textContent = "● Сервер подключен";
  } catch { $("connection").textContent = "○ Сервер недоступен"; running = false; controls(); $("stale").hidden = $("video").hidden; $("stale").textContent = "Соединение потеряно · последний кадр"; }
  setTimeout(poll, 900);
}
async function refreshFrame() {
  if (!frameBusy && latestStatus?.processed_frames > 0 && (running || $("video").hidden) && !document.hidden) {
    frameBusy = true;
    try {
      const response = await fetch("/frame.jpg", {cache:"no-store"});
      if (response.status === 200) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob), previous = $("video").src;
        $("video").src = url; $("video").hidden = false; $("placeholder").hidden = true;
        if (previous.startsWith("blob:")) URL.revokeObjectURL(previous);
      }
    } catch { /* Connection state is reported by the status poll. */ }
    finally { frameBusy = false; }
  }
  setTimeout(refreshFrame, 140);
}
$("date").textContent = new Date().toLocaleDateString("ru-RU", {day:"numeric",month:"long",year:"numeric"});
api("/demo-status").then(status => { demoReady = status.ready; if (!demoReady) message("Для реального демо сначала скачайте видео и модель: python -m src.assets. Нужен пакет onnxruntime."); }).catch(()=>{});
poll(); refreshFrame();
