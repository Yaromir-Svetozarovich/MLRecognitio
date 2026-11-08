/* Blank Aligner — страница нормализации + извлечение полей ответов (ROI) */

const alignForm = document.getElementById('alignForm');
const loading = document.getElementById('loading');
const result = document.getElementById('result');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');
const alignedImg = document.getElementById('alignedImg');
const edgesImg = document.getElementById('edgesImg');
const resultInfo = document.getElementById('resultInfo');

const extractBtn = document.getElementById('extractBtn');
const extractResult = document.getElementById('extractResult');
const extractInfo = document.getElementById('extractInfo');
const cropsGrid = document.getElementById('cropsGrid');

// Сохраняем данные после нормализации, чтобы использовать их при извлечении полей
let alignedImageDataUrl = null;
let templateId = null;

// --- Вспомогательная: загрузить spec.json через /api/templates, вернуть template_id ---
async function uploadSpecFile(specFile) {
    const formData = new FormData();
    formData.append('spec_file', specFile);
    const response = await fetch('/api/templates', {
        method: 'POST',
        body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.detail?.message || 'Не удалось загрузить spec.json');
    }
    return data.template_id;
}

alignForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const specFileInput = document.getElementById('spec_file');
    const specFile = specFileInput.files && specFileInput.files[0] ? specFileInput.files[0] : null;
    const textTemplateId = document.getElementById('template_id').value.trim() || null;

    // Сброс состояния
    result.classList.add('hidden');
    error.classList.add('hidden');
    extractResult.classList.add('hidden');
    extractBtn.style.display = 'none';
    loading.classList.remove('hidden');

    try {
        // Если загружен spec.json файл — регистрируем его на сервере сначала
        if (specFile) {
            templateId = await uploadSpecFile(specFile);
        } else {
            templateId = textTemplateId;
        }

        // Формируем FormData для /api/align (без spec_file, только photo + template_id + corner_mode)
        const alignFormData = new FormData();
        alignFormData.append('photo', alignForm.photo.files[0]);
        if (templateId) {
            alignFormData.append('template_id', templateId);
        }
        const cornerMode = alignForm.corner_mode.value;
        if (cornerMode) {
            alignFormData.append('corner_mode', cornerMode);
        }

        const response = await fetch('/api/align', {
            method: 'POST',
            body: alignFormData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail?.message || 'Ошибка сервера');
        }

        loading.classList.add('hidden');

        if (data.status === 'ok') {
            // Показываем метаданные (без огромных data URL).
            const meta = { ...data };
            delete meta.aligned_image;
            delete meta.edges_image;
            resultInfo.textContent = JSON.stringify(meta, null, 2);

            if (data.aligned_image) {
                alignedImg.src = data.aligned_image;
                alignedImageDataUrl = data.aligned_image;
            }
            if (data.edges_image) {
                edgesImg.src = data.edges_image;
            }

            result.classList.remove('hidden');

            // Кнопка извлечения полей доступна после нормализации.
            if (alignedImageDataUrl) {
                extractBtn.style.display = 'inline-block';
            }
        } else {
            throw new Error(data.message || 'Неизвестная ошибка');
        }
    } catch (err) {
        loading.classList.add('hidden');
        errorMessage.textContent = err.message;
        error.classList.remove('hidden');
    }
});

// --- Извлечение полей ответов (ROI) ---

extractBtn.addEventListener('click', async () => {
    if (!alignedImageDataUrl) return;

    if (!templateId) {
        errorMessage.textContent = 'Укажите ID шаблона или загрузите spec.json перед извлечением полей.';
        error.classList.remove('hidden');
        return;
    }

    // Сброс состояния извлечения
    extractResult.classList.add('hidden');
    extractInfo.textContent = '';
    cropsGrid.innerHTML = '';
    extractBtn.disabled = true;
    extractBtn.textContent = 'Извлекаю…';

    try {
        // Преобразуем data URL обратно в Blob, чтобы отправить на /api/extract
        const response = await fetch(alignedImageDataUrl);
        const blob = await response.blob();

        const formData = new FormData();
        formData.append('photo', blob, 'aligned.jpg');
        formData.append('template_id', templateId);

        const extractResponse = await fetch('/api/extract', {
            method: 'POST',
            body: formData,
        });

        const data = await extractResponse.json();

        if (!extractResponse.ok) {
            throw new Error(data.detail?.message || 'Ошибка извлечения полей');
        }

        if (data.status !== 'ok') {
            throw new Error(data.message || 'Неизвестная ошибка');
        }

        extractInfo.textContent = JSON.stringify({
            template_version: data.template_version,
            num_crops: data.num_crops,
        }, null, 2);

        // Рисуем сетку полей ответов
        for (const crop of data.crops) {
            const item = document.createElement('div');
            item.className = 'preview-item';
            item.style.display = 'flex';
            item.style.flexDirection = 'column';
            item.style.alignItems = 'center';
            item.style.gap = '0.5rem';

            const title = document.createElement('div');
            title.textContent = crop.field_id;
            title.style.fontWeight = '600';
            title.style.fontSize = '0.9rem';

            const img = document.createElement('img');
            img.src = crop.crop_image;
            img.alt = crop.field_id;
            img.style.width = '100%';
            img.style.border = '1px solid var(--border)';
            img.style.borderRadius = '8px';

            const meta = document.createElement('div');
            meta.textContent = `coords: [${crop.coords.join(', ')}]  size: [${crop.size.join(', ')}]`;
            meta.style.fontSize = '0.75rem';
            meta.style.color = 'var(--text-muted)';

            item.appendChild(title);
            item.appendChild(img);
            item.appendChild(meta);
            cropsGrid.appendChild(item);
        }

        extractResult.classList.remove('hidden');
    } catch (err) {
        errorMessage.textContent = err.message;
        error.classList.remove('hidden');
    } finally {
        extractBtn.disabled = false;
        extractBtn.textContent = 'Извлечь поля ответов';
    }
});
