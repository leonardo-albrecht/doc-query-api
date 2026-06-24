/**
 * Doc Query AI — Frontend Application Logic
 * ==========================================
 * Handles PDF upload, chat queries, conversation history,
 * and API communication with the FastAPI backend.
 */

const API_BASE = window.location.origin;

// ── DOM Elements ──────────────────────────────────────────
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const dropzoneContent = document.getElementById('dropzoneContent');
const uploadProgress = document.getElementById('uploadProgress');
const uploadFill = document.getElementById('uploadFill');
const uploadText = document.getElementById('uploadText');
const uploadResult = document.getElementById('uploadResult');
const resultFilename = document.getElementById('resultFilename');
const resultMeta = document.getElementById('resultMeta');

const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');

const historyList = document.getElementById('historyList');
const clearHistory = document.getElementById('clearHistory');

const apiStatusEl = document.getElementById('apiStatus');

// ── State ─────────────────────────────────────────────────
let isUploading = false;
let isQuerying = false;
let documentReady = false;
let conversationHistory = [];

// ── API Health Check ──────────────────────────────────────
async function checkHealth() {
    const dot = apiStatusEl.querySelector('.status-dot');
    const text = apiStatusEl.querySelector('.status-text');

    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
        const data = await res.json();
        dot.classList.add('online');
        dot.classList.remove('offline');
        text.textContent = `Online v${data.version}`;
    } catch {
        dot.classList.add('offline');
        dot.classList.remove('online');
        text.textContent = 'Offline';
    }
}

checkHealth();
setInterval(checkHealth, 30000);

// ── Dropzone Events ───────────────────────────────────────
dropzone.addEventListener('click', () => {
    if (!isUploading) fileInput.click();
});

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].name.toLowerCase().endsWith('.pdf')) {
        handleFileUpload(files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFileUpload(fileInput.files[0]);
    }
});

// ── File Upload ───────────────────────────────────────────
async function handleFileUpload(file) {
    if (isUploading) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showUploadError('Apenas arquivos PDF são aceitos.');
        return;
    }

    isUploading = true;
    uploadResult.style.display = 'none';
    uploadProgress.style.display = 'block';
    uploadFill.style.width = '0%';
    uploadText.textContent = `Enviando ${file.name}...`;

    // Simulate progressive upload
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + Math.random() * 15, 85);
        uploadFill.style.width = `${progress}%`;
    }, 200);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData,
        });

        clearInterval(progressInterval);

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Erro no upload');
        }

        const data = await res.json();

        // Complete progress
        uploadFill.style.width = '100%';
        uploadText.textContent = 'Processamento concluído!';

        setTimeout(() => {
            uploadProgress.style.display = 'none';
            uploadResult.style.display = 'flex';
            resultFilename.textContent = data.filename;
            resultMeta.textContent = `${data.chunks_gerados} chunks · ${data.caracteres_totais.toLocaleString()} caracteres`;
            documentReady = true;
            updateSendButton();
        }, 500);

    } catch (err) {
        clearInterval(progressInterval);
        uploadProgress.style.display = 'none';
        showUploadError(err.message);
    } finally {
        isUploading = false;
        fileInput.value = '';
    }
}

function showUploadError(msg) {
    uploadResult.style.display = 'flex';
    uploadResult.style.background = 'var(--red-dim)';
    uploadResult.style.borderColor = 'rgba(239, 68, 68, 0.2)';
    uploadResult.querySelector('.upload-result__icon').style.background = 'var(--red)';
    resultFilename.textContent = 'Erro no upload';
    resultMeta.textContent = msg;

    setTimeout(() => {
        uploadResult.style.display = 'none';
        uploadResult.style.background = '';
        uploadResult.style.borderColor = '';
        uploadResult.querySelector('.upload-result__icon').style.background = '';
    }, 5000);
}

// ── Chat Input ────────────────────────────────────────────
chatInput.addEventListener('input', () => {
    // Auto-resize textarea
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    updateSendButton();
});

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});

sendBtn.addEventListener('click', sendQuestion);

function updateSendButton() {
    sendBtn.disabled = !chatInput.value.trim() || isQuerying;
}

// ── Send Question ─────────────────────────────────────────
async function sendQuestion() {
    const question = chatInput.value.trim();
    if (!question || isQuerying) return;

    isQuerying = true;
    updateSendButton();

    // Add user message
    appendMessage('user', question);

    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Show typing indicator
    const typingEl = appendTypingIndicator();

    const startTime = performance.now();

    try {
        const res = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, top_k: 3 }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Erro na consulta');
        }

        const data = await res.json();
        const elapsed = Math.round(performance.now() - startTime);

        // Remove typing indicator
        typingEl.remove();

        // Add AI response
        appendMessage('ai', data.answer, {
            sources: data.sources,
            tokens: data.tokens_used,
            latency: elapsed,
        });

        // Add to history
        addToHistory(question, elapsed, data.tokens_used);

    } catch (err) {
        typingEl.remove();
        appendMessage('error', err.message);
    } finally {
        isQuerying = false;
        updateSendButton();
    }
}

// ── Message Rendering ─────────────────────────────────────
function appendMessage(type, content, meta = {}) {
    const msgEl = document.createElement('div');
    msgEl.className = `message message--${type === 'error' ? 'system' : type}`;

    if (type === 'user') {
        msgEl.innerHTML = `
            <div class="message__avatar message__avatar--user">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
            </div>
            <div class="message__content">
                <div class="message__bubble message__bubble--user">${escapeHtml(content)}</div>
            </div>`;
    } else if (type === 'ai') {
        let sourcesHtml = '';
        if (meta.sources && meta.sources.length > 0) {
            const sourceChips = meta.sources.map((s, i) =>
                `<div class="source-chip"><strong>Chunk ${i + 1}</strong>${escapeHtml(s.substring(0, 250))}${s.length > 250 ? '…' : ''}</div>`
            ).join('');

            sourcesHtml = `
                <div class="sources-panel">
                    <button class="sources-toggle" onclick="toggleSources(this)">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
                        ${meta.sources.length} fonte${meta.sources.length > 1 ? 's' : ''} utilizada${meta.sources.length > 1 ? 's' : ''}
                    </button>
                    <div class="sources-list">${sourceChips}</div>
                </div>`;
        }

        let metaHtml = '';
        if (meta.latency || meta.tokens) {
            metaHtml = `<div class="message__meta">`;
            if (meta.latency) {
                metaHtml += `<span class="message__meta-item">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    ${meta.latency}ms
                </span>`;
            }
            if (meta.tokens) {
                metaHtml += `<span class="message__meta-item">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    ${meta.tokens} tokens
                </span>`;
            }
            metaHtml += `</div>`;
        }

        msgEl.innerHTML = `
            <div class="message__avatar message__avatar--ai">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2a4 4 0 0 1 4 4v1a4 4 0 0 1-8 0V6a4 4 0 0 1 4-4z"/><path d="M16 15a4 4 0 0 1 4 4v2H4v-2a4 4 0 0 1 4-4h8z"/><circle cx="12" cy="6" r="1.5" fill="currentColor"/></svg>
            </div>
            <div class="message__content">
                <div class="message__bubble message__bubble--ai">${formatAnswer(content)}</div>
                ${sourcesHtml}
                ${metaHtml}
            </div>`;
    } else if (type === 'error') {
        msgEl.innerHTML = `
            <div class="message__avatar message__avatar--ai">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
            </div>
            <div class="message__content">
                <div class="message__bubble message__bubble--error">${escapeHtml(content)}</div>
            </div>`;
    }

    chatMessages.appendChild(msgEl);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'message message--system';
    el.innerHTML = `
        <div class="message__avatar message__avatar--ai">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2a4 4 0 0 1 4 4v1a4 4 0 0 1-8 0V6a4 4 0 0 1 4-4z"/><path d="M16 15a4 4 0 0 1 4 4v2H4v-2a4 4 0 0 1 4-4h8z"/><circle cx="12" cy="6" r="1.5" fill="currentColor"/></svg>
        </div>
        <div class="message__content">
            <div class="message__bubble message__bubble--ai">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>`;
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return el;
}

// ── Sources Toggle ────────────────────────────────────────
window.toggleSources = function (btn) {
    btn.classList.toggle('open');
    const list = btn.nextElementSibling;
    list.classList.toggle('open');
};

// ── History ───────────────────────────────────────────────
function addToHistory(question, latency, tokens) {
    conversationHistory.unshift({ question, latency, tokens, time: new Date() });

    // Keep max 50 entries
    if (conversationHistory.length > 50) conversationHistory.pop();

    renderHistory();
}

function renderHistory() {
    if (conversationHistory.length === 0) {
        historyList.innerHTML = `<div class="history-empty"><p>Nenhuma consulta ainda</p></div>`;
        return;
    }

    historyList.innerHTML = conversationHistory.map((item, i) => `
        <div class="history-item" onclick="reaskQuestion(${i})">
            <p class="history-item__question">${escapeHtml(item.question)}</p>
            <div class="history-item__meta">
                <span>${item.latency}ms</span>
                <span>${item.tokens} tok</span>
            </div>
        </div>
    `).join('');
}

window.reaskQuestion = function (index) {
    const item = conversationHistory[index];
    if (item && !isQuerying) {
        chatInput.value = item.question;
        chatInput.dispatchEvent(new Event('input'));
        chatInput.focus();
    }
};

clearHistory.addEventListener('click', () => {
    conversationHistory = [];
    renderHistory();
});

// ── Utilities ─────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatAnswer(text) {
    // Convert basic markdown-like formatting
    let html = escapeHtml(text);
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    return html;
}

// ── Initialize ────────────────────────────────────────────
renderHistory();
chatInput.focus();
