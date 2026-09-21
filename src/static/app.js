"use strict";
const $ = (id) => document.getElementById(id);
const emptyAlerts = $("alerts").firstElementChild.cloneNode(true);
const fallbackLabels = {missing_api_key:"не указан ключ API", invalid_api_key:"неверный ключ API", access_denied:"нет доступа к модели", model_not_available:"модель недоступна", provider_rate_limit:"лимит Groq", provider_timeout:"Groq не ответил вовремя", provider_unreachable:"нет соединения с провайдером", invalid_provider_response:"некорректный ответ модели", rate_limited_locally:"интервал между VLM-запросами — 70 секунд", missing_incident_image:"нет снимка инцидента"};
let running = false, busy = false, demoReady = false, lastAlerts = "", frameBusy = false;
let latestStatus = null;
let presets = {};
const notes = {
  demo: "Наблюдение за пешеходами. Количество людей не является основанием для тревоги.",
  traffic: "Дорожная запись. Синяя область — только одна сторона дороги, стрелка — разрешенное направление. Оранжевая область — учебная закрытая зона.",
  "traffic-reversed": "КОНТРОЛИРУЕМЫЙ ТЕСТ: видео воспроизводится в обратном порядке. Это не запись настоящего нарушения ПДД.",
  fight: "Постановочный клип AIRTLab с исследовательской меткой fight. Проверяется возможная агрессия по последовательности кадров; это не настоящая уличная камера.",
  interaction: "Постановочный исследовательский клип AIRTLab в помещении. Экспериментальная проверка движений по 3 кадрам; это не обученный классификатор драк.",
  nonviolent: "AIRTLab: объятия и жесты. Контроль ложных срабатываний; активные движения сами по себе не означают драку.",
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
  const preset = presets[$("source").value];
  if (preset) {
    $("scenario").value = preset.scenario;
    $("direction").value = preset.allowed_direction || "down";
    if (preset.zone) ["left","top","right","bottom"].forEach((id,i)=>$(id).value=preset.zone[i]);
    if (preset.direction_zone) ["dirLeft","dirTop","dirRight","dirBottom"].forEach((id,i)=>$(id).value=preset.direction_zone[i]);
  }
  if (!preset) $("scenario").value = "observe";
  message("");
});
$("controls").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (presets[$("source").value] && !presets[$("source").value].ready) {
    message("Подготовьте реальное демо в терминале: python -m pip install -r requirements-onnx.txt, затем python -m src.assets --scenarios. После этого обновите страницу.", true); return;
  }
  const choice = $("source").value;
  const source = choice === "webcam" ? 0 : choice === "custom" ? $("customSource").value.trim() : choice;
  const zone = ["left", "top", "right", "bottom"].map(id => Number($(id).value));
  if (zone[0] >= zone[2] || zone[1] >= zone[3]) { message("Левая граница должна быть меньше правой, а верхняя — меньше нижней.", true); return; }
  busy = true; controls(); message("Открываем видео и загружаем детектор…");
  try {
    const status = await api("/start-stream", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({source, zone, scenario: $("scenario").value, allowed_direction: $("direction").value, direction_zone: ["dirLeft","dirTop","dirRight","dirBottom"].map(id=>Number($(id).value)), confidence: Number($("confidence").value), dwell_seconds: Number($("dwell").value), cooldown_seconds: Number($("cooldown").value)})});
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
  $("people").textContent = status.objects;
  $("inZone").textContent = status.vehicles;
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
  $("videoLabel").textContent = status.controlled_test ? "КОНТРОЛЬНЫЙ ТЕСТ · ОБРАТНАЯ ЗАПИСЬ" : labels[status.source_kind] || "";
  $("videoLabel").hidden = !status.source_kind;
  if (status.report_mode === "vlm") {
    $("reportMode").textContent = `Groq Vision · ${status.report_model}. До 3 последовательных кадров передаются в Groq, не чаще раза в 70 секунд. ` + (!status.report_key_configured ? "Ключ API не настроен." : status.report_last_error ? "Последняя ошибка: " + (fallbackLabels[status.report_last_error] || status.report_last_error) : "Визуальная оценка + отчет на русском.");
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
    severity.textContent = alert.severity === "critical" ? "СРОЧНАЯ ПРОВЕРКА" : alert.severity === "info" ? "ИНФОРМАЦИЯ" : "ПРОВЕРИТЬ ЭПИЗОД";
    severity.classList.toggle("critical", alert.severity === "critical");
    node.querySelector("time").textContent = new Date(event.timestamp).toLocaleTimeString("ru-RU");
    const names = {person_zone:"Человек в заданной закрытой зоне",wrong_way:"Возможное движение против направления",vehicle_zone_entry:"Заезд в заданную закрытую область",interaction_candidate:"Взаимодействие людей · эксперимент"};
    node.querySelector("h3").textContent = (event.controlled_test ? "Тест · " : "") + (names[event.event_type] || "Эпизод");
    node.querySelector(".alert-action").textContent = alert.action === "none" ? "Оснований для тревоги пока нет." : "Проверьте последовательность кадров и настройку сценария.";
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
      const verdicts = {confirmed:"VLM: признаки события видны", not_confirmed:"VLM: событие не подтверждено", uncertain:"VLM: недостаточно уверенности"};
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
    if (event.evidence_times?.length > 1) {
      const row = document.createElement("div"); row.className="evidence-sequence";
      event.evidence_times.forEach((time,i)=>{const a=document.createElement("a");a.href=`/alerts/${encodeURIComponent(event.id)}/evidence/${i}.jpg`;a.target="_blank";a.rel="noopener";a.textContent=`Кадр ${i+1} · ${time.toFixed(1)}с`;row.append(a);});
      node.querySelector(".report").after(row);
    }
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
api("/demo-status").then(status => { demoReady = status.ready; presets = status.presets || {}; if (!demoReady) message("Для реального демо сначала скачайте видео и модель: python -m src.assets --scenarios. Нужен пакет onnxruntime."); }).catch(()=>{});
poll(); refreshFrame();
