const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const initialScreen = document.getElementById('initial-screen');
const chatWrapper = document.querySelector('.chat-wrapper');
const inputSection = document.getElementById('input-section-initial-location');

let chatStarted = false;
// !!! 중요: Vercel 배포 후 제공되는 실제 URL로 변경하세요 !!!
const API_ENDPOINT_URL = "YOUR_VERCEL_APP_URL/api/ask"; // 예: https://your-project-name.vercel.app/api/ask

// 메시지 추가 함수 (줄바꿈 처리 개선)
function addMessage(text, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender + '-message');

    // XSS 방지를 위해 textContent를 기본으로 사용하고, 줄바꿈만 <br>로 변경
    // 만약 마크다운 등을 지원하려면 검증된 라이브러리(예: marked.js) 사용 필요
    messageElement.innerHTML = text
        .replace(/&/g, "&") // 기본적인 HTML 이스케이프
        .replace(/</g, "<")
        .replace(/>/g, ">")
        .replace(/"/g, """)
        .replace(/'/g, "'")
        .replace(/\n/g, '<br>'); // 줄바꿈 처리

    chatBox.appendChild(messageElement);
    // 메시지 추가 후 즉시 스크롤
    chatBox.scrollTop = chatBox.scrollHeight;
    // 약간의 시간차를 두고 한 번 더 스크롤 (이미지 로딩 등 비동기 컨텐츠 고려)
    setTimeout(() => chatBox.scrollTop = chatBox.scrollHeight, 100);
}


// 로딩 상태 표시/숨김 함수
function showLoading(show = true) {
    let loadingElement = document.getElementById('loading-indicator');
    if (show) {
        if (!loadingElement) {
            loadingElement = document.createElement('div');
            loadingElement.id = 'loading-indicator';
            // 로딩 메시지도 addMessage 함수를 사용해 안전하게 추가
            loadingElement.classList.add('message', 'ai-message', 'loading');
            loadingElement.innerHTML = '답변 생성 중...'; // 직접 HTML 설정 대신
            chatBox.appendChild(loadingElement);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        loadingElement.style.display = 'block'; // 보이도록 설정
        sendButton.disabled = true;
        userInput.disabled = true;
    } else {
        if (loadingElement) {
            loadingElement.remove(); // 로딩 끝나면 제거
        }
        sendButton.disabled = false;
        userInput.disabled = false;
        // 입력창이 활성화될 때 다시 포커스 (모바일에서는 키보드가 다시 올라올 수 있음)
        // 데스크탑 환경에서는 유용
        if (window.innerWidth > 768) { // 예: 모바일 제외
             userInput.focus();
        }
    }
}

// 메시지 전송 및 API 호출 함수
async function sendMessage() {
    const messageText = userInput.value.trim();
    if (messageText === '' || sendButton.disabled) { // 전송 중일 때(비활성화 시) 중복 전송 방지
        return;
    }

    // 첫 메시지 처리
    if (!chatStarted) {
        chatWrapper.appendChild(inputSection);
        initialScreen.style.display = 'none';
        chatBox.style.display = 'flex';
        chatStarted = true;
    }

    addMessage(messageText, 'user'); // 사용자 메시지 표시
    const currentInput = userInput.value; // 실패 시 복구용
    userInput.value = ''; // 입력창 비우기
    showLoading(true); // 로딩 시작

    try {
        console.log("API 요청 시작:", API_ENDPOINT_URL);
        const response = await fetch(API_ENDPOINT_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: messageText })
        });
        console.log("API 응답 수신:", response.status);

        // 로딩 상태는 성공/실패와 관계없이 여기서 숨김
        showLoading(false);

        if (!response.ok) {
            // 서버에서 에러 응답 (JSON 형태일 수 있음)
            let errorMsg = `서버 응답 오류 (코드: ${response.status})`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.error || errorMsg; // 서버가 보낸 에러 메시지 사용
            } catch (e) {
                // JSON 파싱 실패 시 기본 메시지 사용
                console.error("Error parsing error response:", e);
            }
            console.error('API Error:', errorMsg);
            addMessage(`죄송합니다. 답변 처리 중 오류가 발생했습니다: ${errorMsg}`, 'ai');
            userInput.value = currentInput; // 입력 내용 복원
            return;
        }

        // 성공 응답 처리
        const data = await response.json();
        if (data.answer) {
            addMessage(data.answer, 'ai'); // AI 답변 표시
        } else {
            console.error('API Error: Invalid response format', data);
            addMessage('죄송합니다. 서버로부터 예상치 못한 형식의 응답을 받았습니다.', 'ai');
            userInput.value = currentInput; // 입력 내용 복원
        }

    } catch (error) {
        // 네트워크 오류 또는 fetch 자체의 문제
        showLoading(false); // 로딩 숨김
        console.error('Fetch Error:', error);
        let errorText = '죄송합니다. 서버에 연결할 수 없습니다.';
        if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
             errorText += ' 네트워크 연결을 확인하거나 API 엔드포인트 URL이 정확한지 확인해주세요.';
        } else {
             errorText += ` (${error.message || '알 수 없는 오류'})`;
        }
        addMessage(errorText, 'ai');
        userInput.value = currentInput; // 입력 내용 복원
    }
}

// 이벤트 리스너
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', function(event) {
    // Enter 키 입력 시 + Shift 키 누르지 않았을 때만 전송 (Shift+Enter는 줄바꿈)
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault(); // 기본 동작(줄바꿈 등) 방지
        sendMessage();
    }
});