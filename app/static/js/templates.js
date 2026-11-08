/* Blank Aligner — страница шаблонов + генерация работы из CSV */

document.addEventListener('DOMContentLoaded', async () => {
    loadTemplates();

    const form = document.getElementById('generateWorkForm');
    const loading = document.getElementById('genLoading');
    const result = document.getElementById('genResult');
    const error = document.getElementById('genError');
    const errorMessage = document.getElementById('genErrorMessage');
    const resultTitle = document.getElementById('genResultTitle');
    const resultFiles = document.getElementById('genResultFiles');
    const genDownloadZipBtn = document.getElementById('genDownloadZipBtn');

    // Copies mode toggle
    const copiesModeRadios = document.querySelectorAll('input[name="copies_mode"]');
    const copiesAllGroup = document.getElementById('copiesAllGroup');
    const copiesPerVariantGroup = document.getElementById('copiesPerVariantGroup');
    const copiesVariantsList = document.getElementById('copiesVariantsList');
    const numVariantsInput = document.getElementById('num_variants');

    function updateCopiesUI() {
        const mode = document.querySelector('input[name="copies_mode"]:checked').value;
        if (mode === 'all') {
            copiesAllGroup.style.display = '';
            copiesPerVariantGroup.style.display = 'none';
        } else {
            copiesAllGroup.style.display = 'none';
            copiesPerVariantGroup.style.display = '';
            renderCopiesVariants();
        }
    }

    function renderCopiesVariants() {
        const n = parseInt(numVariantsInput.value) || 1;
        copiesVariantsList.innerHTML = '';
        for (let i = 1; i <= n; i++) {
            const label = document.createElement('label');
            label.style.display = 'inline-block';
            label.style.marginRight = '1rem';
            label.innerHTML = `<input type="number" name="copies_variant_${i}" value="1" min="1" max="50" style="width:6rem;"> вариант ${i}`;
            copiesVariantsList.appendChild(label);
        }
    }

    copiesModeRadios.forEach(r => r.addEventListener('change', updateCopiesUI));
    numVariantsInput.addEventListener('change', renderCopiesVariants);
    updateCopiesUI();

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);

        result.classList.add('hidden');
        error.classList.add('hidden');
        loading.classList.remove('hidden');

        try {
            const response = await fetch('/api/templates/generate-work', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail?.message || 'Ошибка сервера');
            }

            loading.classList.add('hidden');

            resultTitle.textContent = `Работа "${data.work_name}" создана`;
            resultFiles.innerHTML = '';

            if (data.files && data.files.length > 0) {
                const ul = document.createElement('ul');
                ul.style.listStyle = 'none';
                ul.style.paddingLeft = '0';
                data.files.forEach(f => {
                    const li = document.createElement('li');
                    li.style.marginBottom = '0.5rem';
                    li.innerHTML = `<a href="/api/templates/download/${f.path}" style="color: var(--primary); text-decoration: none;">${f.name}</a>`;
                    ul.appendChild(li);
                });
                resultFiles.appendChild(ul);
            }

            if (data.zip_url) {
                genDownloadZipBtn.href = data.zip_url;
                genDownloadZipBtn.style.display = 'inline-block';
            } else {
                genDownloadZipBtn.style.display = 'none';
            }

            result.classList.remove('hidden');
        } catch (err) {
            loading.classList.add('hidden');
            errorMessage.textContent = err.message;
            error.classList.remove('hidden');
        }
    });
});

async function loadTemplates() {
    const list = document.getElementById('templatesList');
    try {
        const response = await fetch('/api/templates');
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail?.message || 'Не удалось загрузить шаблоны');
        }

        list.innerHTML = '';

        if (data.templates && data.templates.length > 0) {
            data.templates.forEach(t => {
                const item = document.createElement('div');
                item.className = 'template-item';
                item.innerHTML = `
                    <div>
                        <strong>${t.template_id || t.spec.template_version}</strong><br>
                        <small>Canvas: ${t.spec.canvas_size?.[0]}x${t.spec.canvas_size?.[1]}, Маркеры: ${t.spec.expected_marker_ids?.length || 0}</small>
                    </div>
                    <a href="/api/templates/${t.template_id}/docx" class="btn" style="padding:0.4rem 0.8rem; font-size:0.9rem;">DOCX</a>
                `;
                list.appendChild(item);
            });
        } else {
            list.innerHTML = '<p class="loading-text">Шаблоны не найдены. Сгенерируйте через форму выше.</p>';
        }
    } catch (err) {
        list.innerHTML = `<p class="loading-text" style="color: var(--error);">Ошибка: ${err.message}</p>`;
    }
}