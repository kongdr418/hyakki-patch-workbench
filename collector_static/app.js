const state = {
  root: '',
  classes: [],
  split: 'train',
  frames: [],
  index: -1,
  image: new Image(),
  boxes: [],
  legacy: [],
  legacyByImage: {},
  selectedLegacy: -1,
  showLegacy: true,
  selectedBox: -1,
  drawing: null,
  trainTimer: null,
  trainJobWasRunning: false,
  selectedImages: new Set(),
  trainPythonOptions: {},
  patchModel: {exists: false, path: '', pt_exists: false, pt_path: ''},
  legacyModel: {available: true, error: ''},
  oas: {configs: [], default_config: '', root: '', config_dir: ''}
};

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const RARITY_ORDER = ['sp', 'ssr', 'sr', 'r', 'n', 'g', 'buff'];
const RARITY_NAMES = {
  sp: 'SP',
  ssr: 'SSR',
  sr: 'SR',
  r: 'R',
  n: 'N',
  g: '呱',
  buff: 'BUFF',
  other: '其他'
};

function sleep(ms) {
  return new Promise(resolve => window.setTimeout(resolve, ms));
}

function setStatus(text) {
  $('status').textContent = text || '';
}

function oasConfigName() {
  const input = $('configInput');
  const value = (input.value || '').trim();
  if (value) return value;
  const fallback = state.oas.default_config || state.oas.configs?.[0] || '';
  if (!fallback) throw new Error('没有找到 OAS 配置，请先在 OAS 创建配置，或手动输入配置名。');
  input.value = fallback;
  return fallback;
}

function renderOasConfigs() {
  const input = $('configInput');
  const list = $('oasConfigList');
  if (!input || !list) return;
  const current = (input.value || '').trim();
  list.innerHTML = '';
  for (const name of state.oas.configs || []) {
    const option = document.createElement('option');
    option.value = name;
    list.appendChild(option);
  }
  if (!current && state.oas.default_config) {
    input.value = state.oas.default_config;
  }
}

function detectSourceName(source = $('detectModelInput')?.value || 'legacy') {
  if (source === 'patch') return '训练模型';
  if (source === 'both') return '双模型';
  return 'OAS 原模型';
}

function detectSourceBadge(source) {
  if (source === 'patch') return '训练';
  if (source === 'legacy') return 'OAS';
  return source || '模型';
}

function warningSuffix(warnings) {
  if (!warnings || warnings.length === 0) return '';
  const lines = warnings.slice(0, 3);
  const more = warnings.length > 3 ? `\n... 还有 ${warnings.length - 3} 条警告` : '';
  return `\n${lines.join('\n')}${more}`;
}

function updateDetectModelOptions() {
  const select = $('detectModelInput');
  if (!select) return;
  const legacyOption = select.querySelector('option[value="legacy"]');
  const patchOption = select.querySelector('option[value="patch"]');
  const bothOption = select.querySelector('option[value="both"]');
  const hasLegacyModel = state.legacyModel?.available !== false;
  const hasPatchModel = Boolean(state.patchModel?.predict_exists ?? state.patchModel?.exists);
  if (legacyOption) {
    legacyOption.disabled = !hasLegacyModel;
    legacyOption.textContent = hasLegacyModel ? 'OAS 原模型' : 'OAS 原模型（不可用）';
  }
  if (patchOption) {
    patchOption.disabled = !hasPatchModel;
    patchOption.textContent = hasPatchModel ? '训练模型（刚生成）' : '训练模型（未生成）';
  }
  if (bothOption) {
    bothOption.disabled = !hasLegacyModel || !hasPatchModel;
    bothOption.textContent = hasLegacyModel && hasPatchModel ? '原模型 + 训练模型' : '原模型 + 训练模型（不可用）';
  }
  if (select.value === 'both' && (!hasLegacyModel || !hasPatchModel)) {
    select.value = hasPatchModel ? 'patch' : 'legacy';
  }
  if (select.value === 'patch' && !hasPatchModel) {
    select.value = 'legacy';
  }
  if (select.value === 'legacy' && !hasLegacyModel && hasPatchModel) {
    select.value = 'patch';
  }
}

function classRarity(item) {
  const rarity = (item.rarity || String(item.label || '').split('_', 1)[0] || 'other').toLowerCase();
  return RARITY_ORDER.includes(rarity) ? rarity : 'other';
}

function labelNumber(label) {
  const match = String(label || '').match(/_(\d+)$/);
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function sortedClasses() {
  return [...state.classes].sort((a, b) => {
    const rarityA = classRarity(a);
    const rarityB = classRarity(b);
    const orderA = RARITY_ORDER.includes(rarityA) ? RARITY_ORDER.indexOf(rarityA) : RARITY_ORDER.length;
    const orderB = RARITY_ORDER.includes(rarityB) ? RARITY_ORDER.indexOf(rarityB) : RARITY_ORDER.length;
    if (orderA !== orderB) return orderA - orderB;
    const numberA = labelNumber(a.label);
    const numberB = labelNumber(b.label);
    if (numberA !== numberB) return numberA - numberB;
    return String(a.label).localeCompare(String(b.label), 'zh-Hans-CN');
  });
}

function appendClassOptions(select, classes) {
  let currentGroup = null;
  let groupElement = null;
  for (const item of classes) {
    const group = classRarity(item);
    if (group !== currentGroup) {
      currentGroup = group;
      groupElement = document.createElement('optgroup');
      groupElement.label = RARITY_NAMES[group] || group.toUpperCase();
      select.appendChild(groupElement);
    }
    const option = document.createElement('option');
    option.value = item.label;
    option.textContent = `${item.label} · ${item.name}`;
    groupElement.appendChild(option);
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {'Content-Type': 'application/json'},
    ...options
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

async function loadState() {
  const data = await api('/api/state');
  state.root = data.root;
  state.classes = data.classes;
  state.patchModel = data.patch_model || {exists: false, path: ''};
  state.legacyModel = data.legacy_model || {available: true, error: ''};
  state.oas = data.oas || {configs: [], default_config: '', root: '', config_dir: ''};
  $('rootInput').value = state.root;
  renderOasConfigs();
  renderClasses();
  updateDetectModelOptions();
  renderLegacyList();
  await loadFrames(state.split);
  updateTrainStatus().catch(error => setStatus(error.message));
}

function renderClasses() {
  const select = $('classSelect');
  const oldSelect = select ? select.value : '';
  select.innerHTML = '';
  const filter = $('frameLabelFilter');
  const oldFilter = filter ? filter.value : '';
  const classes = sortedClasses();
  if (filter) {
    filter.innerHTML = '<option value="">全部标签</option>';
  }
  appendClassOptions(select, classes);
  if (filter) {
    appendClassOptions(filter, classes);
  }
  if (filter && Array.from(filter.options).some(option => option.value === oldFilter)) {
    filter.value = oldFilter;
  }
  if (Array.from(select.options).some(option => option.value === oldSelect)) {
    select.value = oldSelect;
  }
}

async function loadFrames(split, preferredImage = null) {
  const splitChanged = state.split !== split;
  state.split = split;
  if (splitChanged) state.selectedImages.clear();
  const data = await api(`/api/frames?split=${split}`);
  state.frames = data.frames;
  if (state.frames.length === 0) {
    state.index = -1;
    state.boxes = [];
    state.legacy = [];
    state.selectedLegacy = -1;
    renderFrameList();
    renderLegacyList();
    draw();
    $('currentFrame').textContent = '未选择图片';
    return;
  }
  if (preferredImage) {
    const preferredIndex = state.frames.findIndex(frame => frame.image === preferredImage);
    if (preferredIndex >= 0) state.index = preferredIndex;
  }
  if (state.index < 0 || state.index >= state.frames.length) state.index = 0;
  renderFrameList();
  await openFrame(state.index);
}

function classNameForLabel(label) {
  const item = state.classes.find(entry => entry.label === label);
  return item ? item.name : label;
}

function frameMatchesFilter(frame) {
  const labelFilter = $('frameLabelFilter')?.value || '';
  const query = ($('frameSearchInput')?.value || '').trim().toLowerCase();
  const onlyLabeled = $('onlyLabeledInput')?.checked || false;
  const onlyUnlabeled = $('onlyUnlabeledInput')?.checked || false;
  const onlyUntrained = $('onlyUntrainedInput')?.checked || false;
  if (onlyLabeled && frame.boxes.length === 0) return false;
  if (onlyUnlabeled && frame.boxes.length > 0) return false;
  if (onlyUntrained && frame.trained) return false;
  if (labelFilter && !frame.boxes.some(box => box.label === labelFilter)) return false;
  if (!query) return true;

  const haystack = [
    frame.name,
    frame.image,
    ...frame.boxes.flatMap(box => [box.label, classNameForLabel(box.label)])
  ].join(' ').toLowerCase();
  return haystack.includes(query);
}

function filteredFrameEntries() {
  return state.frames
    .map((frame, index) => ({frame, index}))
    .filter(item => frameMatchesFilter(item.frame));
}

function renderFrameList() {
  const list = $('frameList');
  list.innerHTML = '';
  const entries = filteredFrameEntries();
  entries.forEach(({frame, index}, visibleIndex) => {
    const item = document.createElement('div');
    item.className = `frame-item${index === state.index ? ' active' : ''}`;
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = state.selectedImages.has(frame.image);
    checkbox.onchange = (event) => {
      event.stopPropagation();
      if (checkbox.checked) {
        state.selectedImages.add(frame.image);
      } else {
        state.selectedImages.delete(frame.image);
      }
      renderFrameSummary(entries.length);
    };

    const title = document.createElement('button');
    title.type = 'button';
    title.className = 'frame-open';
    const trainMark = frame.trained ? ' · 已训' : (frame.boxes.length ? ' · 未训' : ' · 未标注');
    title.textContent = `${visibleIndex + 1}. ${frame.name} (${frame.boxes.length})${trainMark}`;
    title.onclick = () => openFrame(index);

    item.appendChild(checkbox);
    item.appendChild(title);
    list.appendChild(item);
  });
  renderFrameSummary(entries.length);
}

function renderFrameSummary(visibleCount = filteredFrameEntries().length) {
  const selectedCount = state.selectedImages.size;
  const untrained = state.frames.filter(frame => !frame.trained).length;
  $('frameSummary').textContent = `${state.split}: ${visibleCount}/${state.frames.length} 张，未训练 ${untrained} 张，已选 ${selectedCount} 张`;
  $('moveTrainBtn').disabled = selectedCount === 0 || state.split === 'train';
  $('moveValBtn').disabled = selectedCount === 0 || state.split === 'val';
  $('deleteFramesBtn').disabled = selectedCount === 0;
}

function frameFromImageRel(image) {
  return {
    image,
    name: image.split('/').pop() || image,
    boxes: [],
    mtime: Date.now() / 1000,
    trained: false,
    trained_at: '',
    last_train_run: ''
  };
}

async function openFrame(index) {
  if (index < 0 || index >= state.frames.length) return;
  state.index = index;
  const frame = state.frames[index];
  state.boxes = frame.boxes.map(box => ({...box}));
  state.legacy = state.legacyByImage[frame.image] || [];
  state.selectedBox = -1;
  state.selectedLegacy = -1;
  $('currentFrame').textContent = frame.image;
  await new Promise((resolve, reject) => {
    state.image = new Image();
    state.image.onload = resolve;
    state.image.onerror = reject;
    state.image.src = `/api/image?image=${encodeURIComponent(frame.image)}&v=${frame.mtime}`;
  });
  canvas.width = state.image.naturalWidth;
  canvas.height = state.image.naturalHeight;
  renderFrameList();
  renderBoxList();
  renderLegacyList();
  draw();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.image && state.image.complete && state.image.naturalWidth) {
    ctx.drawImage(state.image, 0, 0);
  } else {
    ctx.fillStyle = '#0b1020';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  if (state.showLegacy) {
    state.legacy.forEach((box, index) => drawLegacyBox(box, index === state.selectedLegacy));
  }
  state.boxes.forEach((box, index) => drawBox(box, index === state.selectedBox));
  if (state.drawing) drawBox(state.drawing, true);
}

function colorForLabel(label) {
  let hash = 0;
  for (const char of label) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return `hsl(${hash % 360}, 78%, 48%)`;
}

function drawBox(box, selected) {
  const color = colorForLabel(box.label);
  ctx.lineWidth = selected ? 4 : 2;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.font = '16px Segoe UI';
  const label = box.label;
  const width = ctx.measureText(label).width + 10;
  ctx.fillRect(box.x, Math.max(0, box.y - 22), width, 22);
  ctx.fillStyle = '#fff';
  ctx.fillText(label, box.x + 5, Math.max(16, box.y - 6));
}

function drawLegacyBox(box, selected) {
  const color = colorForLabel(box.label);
  ctx.save();
  ctx.setLineDash(selected ? [12, 4] : [7, 5]);
  ctx.lineWidth = selected ? 4 : 2;
  ctx.strokeStyle = color;
  ctx.globalAlpha = selected ? 1 : 0.82;
  ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.setLineDash([]);
  ctx.font = '15px Segoe UI';
  const label = `${detectSourceBadge(box.source)} ${box.label} ${Math.round(box.conf * 100)}%`;
  const width = ctx.measureText(label).width + 10;
  const y = Math.min(canvas.height - 22, box.y + box.h + 2);
  ctx.fillStyle = color;
  ctx.fillRect(box.x, y, width, 22);
  ctx.fillStyle = '#fff';
  ctx.fillText(label, box.x + 5, y + 16);
  ctx.restore();
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height
  };
}

function hitTest(point) {
  for (let i = state.boxes.length - 1; i >= 0; i--) {
    const box = state.boxes[i];
    if (point.x >= box.x && point.x <= box.x + box.w && point.y >= box.y && point.y <= box.y + box.h) {
      return i;
    }
  }
  return -1;
}

canvas.addEventListener('mousedown', (event) => {
  if (state.index < 0) return;
  const point = canvasPoint(event);
  const hit = hitTest(point);
  if (hit >= 0) {
    state.selectedBox = hit;
    renderBoxList();
    draw();
    return;
  }
  const label = $('classSelect').value;
  if (!label) {
    setStatus('先添加或选择一个式神标签');
    return;
  }
  state.drawing = {label, x: point.x, y: point.y, w: 0, h: 0, startX: point.x, startY: point.y};
});

canvas.addEventListener('mousemove', (event) => {
  if (!state.drawing) return;
  const point = canvasPoint(event);
  const x1 = Math.min(state.drawing.startX, point.x);
  const y1 = Math.min(state.drawing.startY, point.y);
  const x2 = Math.max(state.drawing.startX, point.x);
  const y2 = Math.max(state.drawing.startY, point.y);
  Object.assign(state.drawing, {x: x1, y: y1, w: x2 - x1, h: y2 - y1});
  draw();
});

window.addEventListener('mouseup', () => {
  if (!state.drawing) return;
  const box = state.drawing;
  delete box.startX;
  delete box.startY;
  if (box.w > 6 && box.h > 6) {
    state.boxes.push(box);
    state.selectedBox = state.boxes.length - 1;
  }
  state.drawing = null;
  renderBoxList();
  draw();
});

function renderBoxList() {
  const list = $('boxList');
  list.innerHTML = '';
  state.boxes.forEach((box, index) => {
    const item = document.createElement('div');
    item.className = `box-item${index === state.selectedBox ? ' active' : ''}`;
    item.textContent = `${index + 1}. ${box.label} ${Math.round(box.w)}x${Math.round(box.h)}`;
    item.onclick = () => {
      state.selectedBox = index;
      renderBoxList();
      draw();
    };
    list.appendChild(item);
  });
}

function sameAnnotationBox(left, right) {
  return left.label === right.label
    && Math.abs(left.x - right.x) < 2
    && Math.abs(left.y - right.y) < 2
    && Math.abs(left.w - right.w) < 2
    && Math.abs(left.h - right.h) < 2;
}

function boxesFromDetections(detections) {
  const boxes = [];
  for (const box of detections || []) {
    const adopted = {label: box.label, x: box.x, y: box.y, w: box.w, h: box.h};
    if (!boxes.some(existing => sameAnnotationBox(existing, adopted))) {
      boxes.push(adopted);
    }
  }
  return boxes;
}

function adoptAllRecognitionLabels() {
  if (!state.legacy.length) {
    setStatus('当前图还没有模型识别框');
    return;
  }
  let added = 0;
  let skipped = 0;
  for (const adopted of boxesFromDetections(state.legacy)) {
    if (state.boxes.some(existing => sameAnnotationBox(existing, adopted))) {
      skipped += 1;
      continue;
    }
    state.boxes.push(adopted);
    added += 1;
  }
  if (added > 0) {
    state.selectedBox = state.boxes.length - 1;
  }
  renderBoxList();
  draw();
  setStatus(`已按识别标签采用 ${added} 个参考框${skipped ? `，跳过 ${skipped} 个重复框` : ''}。记得点“保存标注”。`);
}

function renderLegacyList() {
  const list = $('legacyList');
  if (!list) return;
  list.innerHTML = '';
  if (state.legacy.length > 0) {
    const actions = document.createElement('div');
    actions.className = 'legacy-actions';
    const adoptAll = document.createElement('button');
    adoptAll.className = 'mini-btn';
    adoptAll.textContent = '全部按识别标签采用';
    adoptAll.onclick = adoptAllRecognitionLabels;
    actions.appendChild(adoptAll);
    list.appendChild(actions);
  }
  state.legacy.forEach((box, index) => {
    const item = document.createElement('div');
    item.className = `legacy-item${index === state.selectedLegacy ? ' active' : ''}`;

    const meta = document.createElement('div');
    meta.className = 'legacy-meta';
    meta.textContent = `${index + 1}. ${detectSourceBadge(box.source)} · ${box.label} · ${box.name} ${Math.round(box.conf * 100)}%`;
    meta.onclick = () => {
      state.selectedLegacy = index;
      renderLegacyList();
      draw();
    };

    const detail = document.createElement('div');
    detail.className = 'legacy-detail';
    detail.textContent = `${Math.round(box.w)}x${Math.round(box.h)} · ${box.rarity.toUpperCase()}`;

    const adopt = document.createElement('button');
    adopt.className = 'mini-btn';
    adopt.textContent = '按当前标签采用';
    adopt.onclick = (event) => {
      event.stopPropagation();
      const label = $('classSelect').value;
      if (!label) {
        setStatus('先添加或选择一个要训练的新式神标签');
        return;
      }
      state.boxes.push({label, x: box.x, y: box.y, w: box.w, h: box.h});
      state.selectedBox = state.boxes.length - 1;
      renderBoxList();
      draw();
      setStatus(`已用 ${label} 采用参考框`);
    };

    item.appendChild(meta);
    item.appendChild(detail);
    item.appendChild(adopt);
    list.appendChild(item);
  });
}

async function saveAnnotations() {
  if (state.index < 0) return;
  const frame = state.frames[state.index];
  const data = await saveImageAnnotations(frame.image, state.boxes);
  const addedText = data.added_classes && data.added_classes.length
    ? `，自动加入 ${data.added_classes.length} 个 OAS 标签`
    : '';
  setStatus(`已保存 ${data.boxes.length} 个框${addedText}`);
  renderFrameList();
  renderBoxList();
  draw();
}

async function saveImageAnnotations(image, boxes) {
  const data = await api('/api/annotations', {
    method: 'POST',
    body: JSON.stringify({image, boxes})
  });
  if (data.classes) {
    state.classes = data.classes;
    renderClasses();
  }
  const frame = state.frames.find(item => item.image === image);
  if (frame) {
    frame.boxes = data.boxes;
    frame.trained = false;
    frame.trained_at = '';
    frame.last_train_run = '';
  }
  if (state.frames[state.index]?.image === image) {
    state.boxes = data.boxes.map(box => ({...box}));
  }
  return data;
}

function legacySettings() {
  return {
    source: $('detectModelInput')?.value || 'legacy',
    conf_threshold: Number($('legacyConfInput').value || 0.25),
    iou_threshold: Number($('legacyIouInput').value || 0.7)
  };
}

function setLegacyForImage(image, detections) {
  state.legacyByImage[image] = detections || [];
  const frame = state.frames[state.index];
  if (frame && frame.image === image) {
    state.legacy = state.legacyByImage[image];
    state.selectedLegacy = -1;
    renderLegacyList();
    draw();
  }
}

async function predictCurrentLegacy() {
  if (state.index < 0) {
    setStatus('先选择一张图片');
    return {detections: []};
  }
  const frame = state.frames[state.index];
  const data = await predictImageLegacy(frame.image);
  setStatus(`${detectSourceName(data.source)}识别 ${data.detections.length} 个参考框${warningSuffix(data.warnings)}`);
  return data;
}

async function predictImageLegacy(image) {
  const data = await api('/api/legacy-detect', {
    method: 'POST',
    body: JSON.stringify({image, ...legacySettings()})
  });
  setLegacyForImage(image, data.detections);
  return data;
}

async function predictBatchLegacy(images = null) {
  const data = await api('/api/legacy-detect-batch', {
    method: 'POST',
    body: JSON.stringify({split: state.split, images, ...legacySettings()})
  });
  for (const [image, detections] of Object.entries(data.predictions)) {
    setLegacyForImage(image, detections);
  }
  setStatus(`${detectSourceName(data.source)}预识别 ${data.images} 张，${data.detections} 个参考框${warningSuffix(data.warnings)}`);
  return data;
}

function chunkItems(items, size) {
  const chunks = [];
  for (let i = 0; i < items.length; i += size) {
    chunks.push(items.slice(i, i + size));
  }
  return chunks;
}

async function autoAnnotateUnlabeledAndExport() {
  const targets = state.frames
    .filter(frame => frame.boxes.length === 0)
    .map(frame => frame.image);
  if (!targets.length) {
    setStatus(`${state.split} 当前没有未标注图片`);
    return;
  }

  const sourceName = detectSourceName();
  const chunks = chunkItems(targets, 20);
  const predictions = {};
  let detectedBoxes = 0;

  for (let i = 0; i < chunks.length; i += 1) {
    setStatus(`${sourceName}识别未标注图：第 ${i + 1}/${chunks.length} 批，已识别 ${Object.keys(predictions).length}/${targets.length} 张`);
    const data = await api('/api/legacy-detect-batch', {
      method: 'POST',
      body: JSON.stringify({split: state.split, images: chunks[i], ...legacySettings()})
    });
    for (const [image, detections] of Object.entries(data.predictions || {})) {
      predictions[image] = detections || [];
      setLegacyForImage(image, detections || []);
      detectedBoxes += (detections || []).length;
    }
    if (data.warnings && data.warnings.length) {
      setStatus(`${sourceName}识别未标注图：第 ${i + 1}/${chunks.length} 批完成${warningSuffix(data.warnings)}`);
    }
  }

  let savedImages = 0;
  let savedBoxes = 0;
  let skippedImages = 0;
  for (const image of targets) {
    const boxes = boxesFromDetections(predictions[image] || []);
    if (!boxes.length) {
      skippedImages += 1;
      continue;
    }
    setStatus(`保存自动标注：${savedImages + skippedImages + 1}/${targets.length} 张，已保存 ${savedImages} 张`);
    const data = await saveImageAnnotations(image, boxes);
    savedImages += 1;
    savedBoxes += data.boxes.length;
  }

  renderFrameList();
  renderBoxList();
  draw();
  const exported = await api('/api/export', {method: 'POST', body: '{}'});
  await updateTrainStatus();
  setStatus(`自动标注完成：识别 ${targets.length} 张，参考框 ${detectedBoxes} 个，保存 ${savedImages} 张/${savedBoxes} 框，跳过 ${skippedImages} 张空识别。已导出配置。\n${exported.data_yaml}\n${exported.patch_labels}`);
}

function trainSettings() {
  return {
    python: $('trainPythonInput').value || null,
    model: $('trainModelInput').value || 'yolov8n.pt',
    mode: $('trainModeInput').value || 'full',
    epochs: Number($('trainEpochsInput').value || 120),
    imgsz: Number($('trainImgszInput').value || 640),
    batch: Number($('trainBatchInput').value || 16),
    device: $('trainDeviceInput').value || 'cpu',
    workers: Number($('trainWorkersInput').value || 4),
    cache: $('trainCacheInput').value || 'ram',
    force: $('trainForceInput').checked,
    archive_existing: $('trainArchiveInput').checked
  };
}

function useCurrentBestModel() {
  if (!state.patchModel?.pt_exists || !state.patchModel?.pt_path) {
    setStatus('还没有生成 best.pt，先完成一次训练后才能继续训练。');
    return;
  }
  $('trainModelInput').value = state.patchModel.pt_path;
  $('trainModeInput').value = 'incremental';
  setStatus('已切换为当前 best.pt，训练方式设为“增量未训练”。');
}

function moduleText(modules) {
  return Object.entries(modules)
    .map(([name, ok]) => `${name}:${ok ? 'OK' : '缺'}`)
    .join('  ');
}

function gpuText(env) {
  const gpu = env.gpu || {hardware: [], usable: false, reason: ''};
  const hardware = gpu.hardware || [];
  const torch = env.torch || {};
  if (!hardware.length) return '显卡：未检测到 NVIDIA';
  const first = hardware[0];
  const memory = first.memory_total_mb ? ` ${Math.round(first.memory_total_mb / 1024)}GB` : '';
  const cuda = torch.cuda_version ? ` CUDA ${torch.cuda_version}` : 'CPU 版 PyTorch';
  const usable = gpu.usable ? '可用' : '不可用';
  return `显卡：GPU ${first.index} ${first.name}${memory} · ${usable} · ${cuda}`;
}

function updateDeviceOptions(env) {
  const gpu = env.gpu || {hardware: [], usable: false};
  const hardware = gpu.hardware || [];
  const option = $('gpuDeviceOption');
  if (hardware.length) {
    const first = hardware[0];
    option.textContent = `GPU ${first.index} · ${first.name}`;
    option.value = String(first.index);
  } else {
    option.textContent = 'GPU';
    option.value = '0';
  }
  option.disabled = !gpu.usable;
  if ($('trainDeviceInput').value !== 'cpu' && !gpu.usable) {
    $('trainDeviceInput').value = 'cpu';
  }
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[char]));
}

function stripAnsi(text) {
  return String(text || '').replace(/\x1b\[[0-9;?]*[A-Za-z]/g, '');
}

function parseTrainProgress(logText) {
  const clean = stripAnsi(logText).replace(/\r/g, '\n');
  const lines = clean.split('\n').map(line => line.trim()).filter(Boolean);
  let epoch = null;
  let totalEpochs = null;
  let batch = null;
  let totalBatches = null;
  let epochLine = '';
  let metricLine = '';

  for (const line of lines) {
    const epochMatch = line.match(/^\s*(\d+)\s*\/\s*(\d+)\s+\S*G\b/);
    const batchMatch = line.match(/:\s*\d+%\s+.*?\s(\d+)\s*\/\s*(\d+)\s/);
    if (epochMatch) {
      epoch = Number(epochMatch[1]);
      totalEpochs = Number(epochMatch[2]);
      epochLine = line;
      if (batchMatch) {
        batch = Number(batchMatch[1]);
        totalBatches = Number(batchMatch[2]);
      }
    }
    if (/^all\s+\d+\s+\d+/.test(line)) {
      metricLine = line;
    }
  }

  let metrics = null;
  if (metricLine) {
    const parts = metricLine.split(/\s+/);
    if (parts.length >= 7) {
      metrics = {
        precision: Number(parts[3]),
        recall: Number(parts[4]),
        map50: Number(parts[5]),
        map5095: Number(parts[6])
      };
    }
  }

  const percent = epoch && totalEpochs ? Math.min(100, Math.round((epoch / totalEpochs) * 100)) : 0;
  return {epoch, totalEpochs, batch, totalBatches, percent, epochLine, metricLine, metrics};
}

function trainProgressHtml(data) {
  const progress = parseTrainProgress(data.log || '');
  const job = data.job || {};
  const hasProgress = progress.epoch && progress.totalEpochs;
  const title = hasProgress
    ? `${progress.epoch} / ${progress.totalEpochs} 轮`
    : (job.running ? '训练中' : (job.exit_code === 0 ? '训练完成' : '暂无训练进度'));
  const batchText = progress.batch && progress.totalBatches
    ? `当前轮 ${progress.batch}/${progress.totalBatches} batch`
    : (progress.epochLine ? escapeHtml(progress.epochLine) : '');
  const metricText = progress.metrics
    ? `P ${progress.metrics.precision.toFixed(3)} · R ${progress.metrics.recall.toFixed(3)} · mAP50 ${progress.metrics.map50.toFixed(3)} · mAP50-95 ${progress.metrics.map5095.toFixed(3)}`
    : (progress.metricLine ? escapeHtml(progress.metricLine) : '');

  return `
    <div class="train-progress-card">
      <div class="train-progress-head">
        <b>${title}</b>
        <span>${job.running ? '运行中' : (job.exit_code === 0 ? '已完成' : '未运行')}</span>
      </div>
      <div class="train-progress-bar"><span style="width: ${progress.percent}%"></span></div>
      <div class="train-progress-meta">${batchText || '等待训练日志'}</div>
      ${metricText ? `<div class="train-progress-meta">${metricText}</div>` : ''}
    </div>
  `;
}

function renderTrainStatus(data) {
  const dataset = data.dataset;
  const env = data.environment;
  const job = data.job;
  const install = data.install || {running: false, exit_code: null};
  state.trainPythonOptions = {
    oas: data.oas_python,
    venv: data.venv_python
  };
  if (!$('trainPythonInput').value) {
    $('trainPythonInput').value = data.default_python;
  }
  updateDeviceOptions(env);
  state.patchModel = data.model || state.patchModel;
  updateDetectModelOptions();
  const useBestBtn = $('useBestModelBtn');
  if (useBestBtn) {
    useBestBtn.disabled = !state.patchModel.pt_exists || job.running || install.running;
    useBestBtn.title = state.patchModel.pt_exists ? state.patchModel.pt_path : '还没有训练生成 best.pt';
  }

  const warnings = dataset.warnings.length
    ? `<div class="warn-text">${dataset.warnings.join('<br>')}</div>`
    : '<div class="ok-text">数据量检查通过</div>';
  const envError = env.error ? `<div class="warn-text">${env.error}</div>` : '';
  const jobText = job.running
    ? '训练中'
    : (job.exit_code === null ? '未启动' : `已结束 ${job.exit_code}`);
  const installText = install.running
    ? '安装中'
    : (install.exit_code === null ? '未启动' : (install.exit_code === 0 ? '已完成' : `失败 ${install.exit_code}`));
  const modelText = data.model.exists ? '已生成' : '未生成';
  const ptText = data.model.pt_exists ? 'best.pt 可用' : 'best.pt 未生成';
  const archiveText = data.model.latest_archive
    ? `最近备份：${escapeHtml(data.model.latest_archive.id)}`
    : '最近备份：无';
  const envText = env.ok ? '可训练' : '缺依赖';
  const commandText = [
    ...data.commands.prepare,
    '',
    '# 如需显卡训练，先安装 CUDA 版 PyTorch：',
    ...(data.commands.cuda_prepare || []),
    '',
    data.commands.train
  ].join('\n');
  const logText = data.log || data.install_log || '';
  const selectedGpuWithoutCuda = $('trainDeviceInput').value !== 'cpu' && !(env.gpu || {}).usable;
  const gpuWarning = (env.gpu && env.gpu.reason && !(env.gpu || {}).usable)
    ? `<div class="warn-text">${escapeHtml(env.gpu.reason)}</div>`
    : '';

  $('trainSummary').innerHTML = `
    ${trainProgressHtml(data)}
    <div class="train-kpis">
      <div><b>${dataset.train.boxes}</b><span>train 框</span></div>
      <div><b>${dataset.val.boxes}</b><span>val 框</span></div>
      <div><b>${dataset.classes.length}</b><span>类别</span></div>
      <div><b>${dataset.train.untrained_images}</b><span>未训练图</span></div>
    </div>
    <div class="train-line">环境：${envText} · ${moduleText(env.modules)}</div>
    <div class="train-line">${escapeHtml(gpuText(env))}</div>
    <div class="train-line">安装：${installText} · 任务：${jobText} · 模型：${modelText} · ${ptText}</div>
    <div class="train-line">${archiveText}</div>
    ${envError}
    ${gpuWarning}
    ${warnings}
    <details class="train-details">
      <summary>准备命令和训练命令</summary>
      <pre class="command-text">${escapeHtml(commandText)}</pre>
    </details>
  `;
  const trainDetails = $('trainSummary').querySelector('.train-details');
  if (trainDetails) trainDetails.open = false;
  $('trainLog').textContent = logText;
  $('trainLog').style.display = logText ? 'block' : 'none';
  $('installDepsBtn').disabled = job.running || install.running || env.ok || !env.exists;
  $('installCudaDepsBtn').disabled = job.running || install.running || !env.exists || !((env.gpu || {}).hardware || []).length || (env.gpu || {}).usable;
  $('trainStartBtn').disabled = job.running || install.running || !env.ok || selectedGpuWithoutCuda;
  $('trainStopBtn').disabled = !job.running;
  const finishedTrainingNow = state.trainJobWasRunning && !job.running;
  state.trainJobWasRunning = job.running;
  if (finishedTrainingNow) {
    loadFrames(state.split).catch(error => setStatus(error.message));
  }

  if ((job.running || install.running) && !state.trainTimer) {
    state.trainTimer = window.setInterval(() => {
      updateTrainStatus().catch(error => setStatus(error.message));
    }, 3000);
  }
  if (!job.running && !install.running && state.trainTimer) {
    window.clearInterval(state.trainTimer);
    state.trainTimer = null;
  }
}

async function updateTrainStatus() {
  const params = new URLSearchParams();
  if ($('trainPythonInput').value) params.set('python', $('trainPythonInput').value);
  params.set('device', $('trainDeviceInput').value || 'cpu');
  const data = await api(`/api/train/status?${params.toString()}`);
  renderTrainStatus(data);
  return data;
}

async function startTraining() {
  const data = await api('/api/train/start', {
    method: 'POST',
    body: JSON.stringify(trainSettings())
  });
  renderTrainStatus(data);
  setStatus('训练已启动，可在训练面板查看日志');
}

async function stopTraining() {
  const data = await api('/api/train/stop', {method: 'POST', body: '{}'});
  renderTrainStatus(data);
  setStatus('已发送停止训练请求');
}

async function installTrainDeps(mode = 'default') {
  const data = await api('/api/train/install-deps', {
    method: 'POST',
    body: JSON.stringify({python: $('trainPythonInput').value || null, mode})
  });
  renderTrainStatus(data);
  setStatus(mode === 'cuda' ? '显卡版 PyTorch 开始安装，日志会持续刷新' : '训练依赖开始安装，日志会持续刷新');
}

function selectVisibleFrames() {
  const entries = filteredFrameEntries();
  for (const {frame} of entries) {
    state.selectedImages.add(frame.image);
  }
  renderFrameList();
}

function clearFrameSelection() {
  state.selectedImages.clear();
  renderFrameList();
}

async function moveSelectedFrames(target) {
  const images = Array.from(state.selectedImages);
  if (images.length === 0) {
    setStatus('先勾选要移动的图片');
    return;
  }
  const data = await api('/api/frames/move', {
    method: 'POST',
    body: JSON.stringify({images, target})
  });
  state.selectedImages.clear();
  await loadFrames(state.split);
  await updateTrainStatus();
  setStatus(`已移动 ${data.moved.length} 张到 ${target}`);
}

async function deleteSelectedFrames() {
  const images = Array.from(state.selectedImages);
  if (images.length === 0) {
    setStatus('先勾选要删除的图片');
    return;
  }
  const preview = images.slice(0, 4).map(item => `- ${item}`).join('\n');
  const more = images.length > 4 ? `\n... 还有 ${images.length - 4} 张` : '';
  if (!window.confirm(`确认删除 ${images.length} 张图片及其标注？\n${preview}${more}`)) {
    return;
  }
  const data = await api('/api/frames/delete', {
    method: 'POST',
    body: JSON.stringify({images})
  });
  state.selectedImages.clear();
  const deletedImages = new Set(data.deleted.map(item => item.image));
  if (state.index >= 0 && deletedImages.has(state.frames[state.index]?.image)) {
    state.index = Math.max(0, state.index - 1);
  }
  await loadFrames(state.split);
  await updateTrainStatus();
  setStatus(`已删除 ${data.deleted.length} 张图片`);
}

async function busy(button, task) {
  const old = button.textContent;
  button.disabled = true;
  try {
    button.textContent = '处理中';
    await task();
  } catch (error) {
    setStatus(error.message);
  } finally {
    button.disabled = false;
    button.textContent = old;
    renderFrameSummary();
  }
}

$('saveRootBtn').onclick = () => busy($('saveRootBtn'), async () => {
  const data = await api('/api/settings', {method: 'POST', body: JSON.stringify({root: $('rootInput').value})});
  state.root = data.root;
  state.classes = data.classes;
  renderClasses();
  await loadFrames(state.split);
});

$('addClassBtn').onclick = () => busy($('addClassBtn'), async () => {
  const data = await api('/api/classes', {
    method: 'POST',
    body: JSON.stringify({
      rarity: $('rarityInput').value,
      label: $('labelInput').value || null,
      name: $('nameInput').value
    })
  });
  state.classes = data.classes;
  renderClasses();
  $('classSelect').value = data.added.label;
  $('labelInput').value = '';
  $('nameInput').value = '';
  setStatus(`已添加 ${data.added.label}`);
});

$('captureBtn').onclick = () => busy($('captureBtn'), async () => {
  const configName = oasConfigName();
  const data = await api('/api/capture', {
    method: 'POST',
    body: JSON.stringify({config_name: configName, split: $('splitInput').value})
  });
  state.split = $('splitInput').value;
  await loadFrames(state.split, data.created[0]);
  setStatus(`采集 ${data.created.length} 张。需要识别时再点“预识别当前图”。`);
});

$('recordBtn').onclick = () => busy($('recordBtn'), async () => {
  const configName = oasConfigName();
  const split = $('splitInput').value;
  const seconds = Math.max(0.1, Number($('recordSecondsInput').value) || 5);
  const interval = Math.max(0.05, Number($('recordIntervalInput').value) || 0.3);
  const planned = Math.max(1, Math.ceil(seconds / interval));
  const created = [];
  if (state.split !== split) {
    await loadFrames(split);
  } else {
    state.split = split;
  }
  for (let i = 0; i < planned; i += 1) {
    setStatus(`连续采集中：正在截第 ${i + 1}/${planned} 张，已保存 ${created.length} 张`);
    const data = await api('/api/capture', {
      method: 'POST',
      body: JSON.stringify({config_name: configName, split, prefix: 'rec'})
    });
    const image = data.created[0];
    created.push(image);
    state.frames.push(frameFromImageRel(image));
    await openFrame(state.frames.length - 1);
    setStatus(`连续采集中：已保存 ${created.length}/${planned} 张`);
    if (i < planned - 1) await sleep(interval * 1000);
  }
  setStatus(`连续采集完成：${created.length} 张。未自动预识别，需要时点“预识别当前图”或“预识别当前分组”。`);
});

$('videoBtn').onclick = () => busy($('videoBtn'), async () => {
  const data = await api('/api/import-video', {
    method: 'POST',
    body: JSON.stringify({
      path: $('videoPathInput').value,
      split: $('splitInput').value,
      interval: Number($('videoIntervalInput').value),
      max_frames: Number($('videoMaxInput').value)
    })
  });
  state.split = $('splitInput').value;
  await loadFrames(state.split);
  setStatus(`抽帧 ${data.created.length} 张。未自动预识别，需要时点“预识别当前图”或“预识别当前分组”。`);
});

$('legacyCurrentBtn').onclick = () => busy($('legacyCurrentBtn'), predictCurrentLegacy);
$('legacyAllBtn').onclick = () => busy($('legacyAllBtn'), () => predictBatchLegacy());
$('autoAnnotateBtn').onclick = () => busy($('autoAnnotateBtn'), autoAnnotateUnlabeledAndExport);
$('detectModelInput').onchange = () => {
  setStatus(`已切换为 ${detectSourceName()} 预识别`);
};
$('showLegacyInput').onchange = () => {
  state.showLegacy = $('showLegacyInput').checked;
  draw();
};
$('trainCheckBtn').onclick = () => busy($('trainCheckBtn'), updateTrainStatus);
$('useOasPythonBtn').onclick = () => {
  if (state.trainPythonOptions.oas) {
    $('trainPythonInput').value = state.trainPythonOptions.oas;
    updateTrainStatus().catch(error => setStatus(error.message));
  }
};
$('useVenvPythonBtn').onclick = () => {
  if (state.trainPythonOptions.venv) {
    $('trainPythonInput').value = state.trainPythonOptions.venv;
    updateTrainStatus().catch(error => setStatus(error.message));
  }
};
$('useBestModelBtn').onclick = useCurrentBestModel;
$('installDepsBtn').onclick = () => busy($('installDepsBtn'), installTrainDeps);
$('installCudaDepsBtn').onclick = () => busy($('installCudaDepsBtn'), () => installTrainDeps('cuda'));
$('trainDeviceInput').onchange = () => updateTrainStatus().catch(error => setStatus(error.message));
$('trainModeInput').onchange = () => {
  if ($('trainModeInput').value === 'incremental' && $('trainModelInput').value === 'yolov8n.pt' && state.patchModel?.pt_exists) {
    $('trainModelInput').value = state.patchModel.pt_path;
  }
};
$('trainStartBtn').onclick = () => busy($('trainStartBtn'), startTraining);
$('trainStopBtn').onclick = () => busy($('trainStopBtn'), stopTraining);
$('frameLabelFilter').onchange = () => {
  if ($('frameLabelFilter').value) {
    $('onlyUnlabeledInput').checked = false;
  }
  renderFrameList();
};
$('frameSearchInput').oninput = renderFrameList;
$('onlyLabeledInput').onchange = () => {
  if ($('onlyLabeledInput').checked) {
    $('onlyUnlabeledInput').checked = false;
  }
  renderFrameList();
};
$('onlyUnlabeledInput').onchange = () => {
  if ($('onlyUnlabeledInput').checked) {
    $('onlyLabeledInput').checked = false;
    $('onlyUntrainedInput').checked = false;
    $('frameLabelFilter').value = '';
  }
  renderFrameList();
};
$('onlyUntrainedInput').onchange = () => {
  if ($('onlyUntrainedInput').checked) {
    $('onlyUnlabeledInput').checked = false;
  }
  renderFrameList();
};
$('selectVisibleBtn').onclick = selectVisibleFrames;
$('clearSelectionBtn').onclick = clearFrameSelection;
$('moveTrainBtn').onclick = () => busy($('moveTrainBtn'), () => moveSelectedFrames('train'));
$('moveValBtn').onclick = () => busy($('moveValBtn'), () => moveSelectedFrames('val'));
$('deleteFramesBtn').onclick = () => busy($('deleteFramesBtn'), deleteSelectedFrames);
$('trainBtn').onclick = () => loadFrames('train');
$('valBtn').onclick = () => loadFrames('val');
$('prevBtn').onclick = () => openFrame(Math.max(0, state.index - 1));
$('nextBtn').onclick = () => openFrame(Math.min(state.frames.length - 1, state.index + 1));
$('deleteBoxBtn').onclick = () => {
  if (state.selectedBox >= 0) {
    state.boxes.splice(state.selectedBox, 1);
    state.selectedBox = -1;
    renderBoxList();
    draw();
  }
};
$('clearBoxesBtn').onclick = () => {
  state.boxes = [];
  state.selectedBox = -1;
  renderBoxList();
  draw();
};
$('saveAnnBtn').onclick = () => busy($('saveAnnBtn'), saveAnnotations);
$('exportBtn').onclick = () => busy($('exportBtn'), async () => {
  const data = await api('/api/export', {method: 'POST', body: '{}'});
  setStatus(`已导出\n${data.data_yaml}\n${data.patch_labels}`);
  await updateTrainStatus();
});

loadState().catch(error => setStatus(error.message));
