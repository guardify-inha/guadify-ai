const API_BASE_URL = 'http://localhost:8000';

// 탭 전환
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        
        // 탭 버튼 활성화
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // 탭 컨텐츠 전환
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
    });
});

// 파일 업로드 영역 이벤트
const fileUploadArea = document.getElementById('file-upload-area');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const analyzeFileBtn = document.getElementById('analyze-file-btn');

fileUploadArea.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        fileName.textContent = file.name;
        fileInfo.style.display = 'flex';
        analyzeFileBtn.style.display = 'block';
        fileUploadArea.style.display = 'none';
    }
});

// 드래그 앤 드롭
fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.classList.add('dragover');
});

fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.classList.remove('dragover');
});

fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadArea.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        const file = e.dataTransfer.files[0];
        fileName.textContent = file.name;
        fileInfo.style.display = 'flex';
        analyzeFileBtn.style.display = 'block';
        fileUploadArea.style.display = 'none';
    }
});

function removeFile() {
    fileInput.value = '';
    fileInfo.style.display = 'none';
    analyzeFileBtn.style.display = 'none';
    fileUploadArea.style.display = 'block';
}

// 텍스트 분석
async function analyzeText() {
    const text = document.getElementById('contract-text').value.trim();
    
    if (!text) {
        alert('계약서 텍스트를 입력해주세요.');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/analyze/text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) {
            throw new Error(`서버 오류: ${response.status}`);
        }
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('분석 오류:', error);
        alert(`분석 중 오류가 발생했습니다: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// 파일 분석
async function analyzeFile() {
    const file = fileInput.files[0];
    
    if (!file) {
        alert('파일을 선택해주세요.');
        return;
    }
    
    showLoading();
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`서버 오류: ${response.status}`);
        }
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('분석 오류:', error);
        alert(`분석 중 오류가 발생했습니다: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// 결과 표시
function displayResults(result) {
    const resultSection = document.getElementById('result-section');
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth' });
    
    // 위험도 평가
    const riskValue = document.getElementById('risk-value');
    const riskBadge = document.querySelector('.risk-badge');
    const summary = document.getElementById('summary');
    
    riskValue.textContent = result.overall_risk_assessment;
    riskBadge.className = 'risk-badge ' + result.overall_risk_assessment.toLowerCase();
    summary.textContent = result.summary;
    
    // 조항별 분석
    const clausesList = document.getElementById('clauses-list');
    clausesList.innerHTML = '';
    
    result.clauses.forEach((clause, index) => {
        const clauseItem = createClauseItem(clause, index);
        clausesList.appendChild(clauseItem);
    });
}

// 조항 아이템 생성
function createClauseItem(clause, index) {
    const div = document.createElement('div');
    div.className = `clause-item ${clause.analysis.is_unfair ? 'unfair' : 'fair'}`;
    
    const badgeClass = clause.analysis.is_unfair ? 'unfair' : 'fair';
    const badgeText = clause.analysis.is_unfair ? '불공정 소지' : '공정';
    
    div.innerHTML = `
        <div class="clause-header">
            <h4>조항 ${index + 1}</h4>
            <span class="clause-badge ${badgeClass}">${badgeText}</span>
        </div>
        
        <div class="clause-original">
            <strong>원본 조항:</strong><br>
            ${clause.original_clause}
        </div>
        
        <div class="clause-analysis">
            <div class="analysis-item">
                <div class="analysis-label">불공정 여부:</div>
                <div class="analysis-content">${clause.analysis.is_unfair ? '불공정 소지 있음' : '공정함'}</div>
            </div>
            
            <div class="analysis-item">
                <div class="analysis-label">이유:</div>
                <div class="analysis-content">${clause.analysis.reason}</div>
            </div>
            
            <div class="analysis-item">
                <div class="analysis-label">약관법 위반 조항:</div>
                <div class="analysis-content">${clause.analysis.evidence_law}</div>
            </div>
        </div>
        
        ${clause.analysis.evidence_law_content ? `
            <div class="clause-law-content">
                <div class="clause-law-content-label">📜 약관법 위반 조항 전체 내용:</div>
                <div class="clause-law-content-text">${clause.analysis.evidence_law_content.replace(/\n/g, '<br>')}</div>
            </div>
        ` : ''}
        
        ${clause.easy_translation ? `
            <div class="clause-translation">
                <div class="clause-translation-label">📝 쉬운 풀이:</div>
                <div>${clause.easy_translation}</div>
            </div>
        ` : ''}
        
        ${clause.suggestion ? `
            <div class="clause-suggestion">
                <div class="clause-suggestion-label">💡 대안 제안:</div>
                <div>${clause.suggestion}</div>
            </div>
        ` : ''}
    `;
    
    return div;
}

// 결과 닫기
function closeResults() {
    document.getElementById('result-section').style.display = 'none';
}

// 로딩 표시
function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

