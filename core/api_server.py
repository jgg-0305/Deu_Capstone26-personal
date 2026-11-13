from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import requests
import json

def _kakao_config_value(key: str, environment_name: str) -> str:
    """Read credentials from the environment or an ignored local JSON file."""
    environment_value = os.environ.get(environment_name)
    if environment_value:
        return environment_value
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_dir, "kakao_local_config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, ValueError):
        return ""
    value = config.get(key, "") if isinstance(config, dict) else ""
    return value if isinstance(value, str) else ""


# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(title="CCTV 이상행동 알림 서버")

# 카카오 인증값은 환경 변수 또는 로컬 설정 파일에서 읽습니다.
KAKAO_ACCESS_TOKEN = _kakao_config_value("access_token", "KAKAO_ACCESS_TOKEN")

KAKAO_REST_API_KEY = _kakao_config_value("rest_api_key", "KAKAO_REST_API_KEY")
KAKAO_REDIRECT_URI = "http://127.0.0.1:8000/kakao/callback"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_FILE = os.path.join(BASE_DIR, "kakao_tokens.json")
ALERT_CONFIDENCE_THRESHOLD = 0.90


def _init_token_store():
    return {"current_user_id": None, "users": {}}


def save_tokens(tokens):
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)


def load_tokens():
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if "users" in data and "current_user_id" in data:
                        return data
                    if "access_token" in data:
                        return {"current_user_id": "default", "users": {"default": data}}
        except Exception:
            pass
    return _init_token_store()


def get_current_user_tokens():
    tokens = load_tokens()
    user_id = tokens.get("current_user_id")
    if user_id is None:
        return {}
    return tokens.get("users", {}).get(str(user_id), {})


def save_user_tokens(user_id, user_tokens):
    tokens = load_tokens()
    users = tokens.setdefault("users", {})
    users[str(user_id)] = user_tokens
    tokens["current_user_id"] = str(user_id)
    save_tokens(tokens)


def refresh_kakao_token():
    tokens = load_tokens()
    current_user_id = tokens.get("current_user_id")
    if not current_user_id:
        print("⚠️ 재발급을 위한 active user가 없습니다. /kakao/login 으로 로그인 해주세요.")
        return False

    user_tokens = tokens.get("users", {}).get(str(current_user_id), {})
    refresh_token = user_tokens.get("refresh_token")
    if not refresh_token:
        print("⚠️ 재발급을 위한 refresh_token이 없습니다. /kakao/login 으로 로그인 해주세요.")
        return False
    
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": KAKAO_REST_API_KEY,
        "refresh_token": refresh_token
    }
    res = requests.post(url, data=data)
    if res.status_code == 200:
        new_tokens = res.json()
        if "refresh_token" not in new_tokens:
            new_tokens["refresh_token"] = refresh_token
        save_user_tokens(current_user_id, new_tokens)
        return True
    else:
        print(f"❌ 토큰 재발급 실패: {res.text}")
        return False

# 알림 기록을 임시로 저장할 리스트 (데이터베이스 역할)
alerts_db = []

# 캡처된 사진을 웹에서 볼 수 있도록 폴더 접근 허용 설정
EVENT_DIR = os.path.join(BASE_DIR, "event_captures")
os.makedirs(EVENT_DIR, exist_ok=True)
app.mount("/event_captures", StaticFiles(directory=EVENT_DIR), name="event_captures")

# 회의때 설계한 JSON 데이터 구조를 파이썬 클래스로 정의 (Pydantic 모델)
class AnomalyEvent(BaseModel):
    event_id: str
    timestamp: str
    anomaly_type: str
    image_path: str
    anomaly_score: float = 0.0

# 서버 기본주소 (GET 방식) - 웹 대시보드 화면
@app.get("/", response_class=HTMLResponse)
def read_root():
    # 가장 최근에 감지된 알림이 위로 오도록 역순 정렬
    recent_alerts = list(reversed(alerts_db))
    
    # 웹 페이지(HTML) 화면 구성
    html_content = """
    <html>
        <head>
            <title>CCTV 실시간 감지 대시보드</title>
            <style>
                body { font-family: 'Malgun Gothic', sans-serif; background-color: #12121e; color: #e0e0f0; padding: 20px; }
                h1 { color: #00d4aa; }
                .alert-card { border: 1px solid #3a3a6a; background-color: #1e1e3e; padding: 15px; margin-bottom: 20px; border-radius: 8px; }
                img { max-width: 400px; border-radius: 4px; margin-top: 10px; border: 2px solid #00d4aa; }
            </style>
            <meta http-equiv="refresh" content="5">
        </head>
        <body>
            <h1> CCTV 실시간 감지 대시보드</h1>
            <hr style="border-color: #3a3a6a;">
    """
    
    if not recent_alerts:
        html_content += "<p>아직 감지된 이상행동이 없습니다. CCTV를 작동시켜주세요.</p>"
    else:
        for alert in recent_alerts:
            filename = os.path.basename(alert.image_path) # 파일명만 추출
            img_url = f"/event_captures/{filename}"       # 웹 이미지 경로 생성
            score_pct = f"{alert.anomaly_score * 100:.1f}%"
            html_content += f"""
            <div class="alert-card">
                <h3>사건 ID: {alert.event_id}</h3>
                <p><b>⏰ 발생시간:</b> {alert.timestamp}</p>
                <p><b>⚠️ 행동유형:</b> {alert.anomaly_type}</p>
                <p><b>✅ 신뢰도:</b> {score_pct}</p>
                <img src="{img_url}" alt="감지 화면">
            </div>
            """
    
    html_content += "</body></html>"
    return html_content

def get_ngrok_url(default_url: str = "http://127.0.0.1:8000") -> str:
    """ngrok 로컬 API를 호출하여 현재 실행 중인 외부 도메인 주소를 반환합니다."""
    try:
        ngrok_res = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=1)
        tunnels = ngrok_res.json().get("tunnels", [])
        for t in tunnels:
            if t.get("public_url", "").startswith("https"):
                return t.get("public_url")
    except requests.exceptions.RequestException:
        print("⚠️ ngrok이 켜져있지 않거나 주소를 획득할 수 없습니다. 기본 로컬 주소를 사용합니다.")
    return default_url

def send_kakao_message(event: AnomalyEvent, is_retry: bool = False):
    """카카오톡 '나에게 보내기' API를 호출하여 알림을 전송합니다."""
    
    active_tokens = get_current_user_tokens()
    access_token = active_tokens.get("access_token") or KAKAO_ACCESS_TOKEN
    
    if not access_token:
        print("⚠️ 사용할 수 있는 액세스 토큰이 없습니다. /kakao/login 으로 다시 로그인 해주세요.")
        return
        
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        public_domain = get_ngrok_url()
        filename = os.path.basename(event.image_path)
        image_public_url = f"{public_domain}/event_captures/{filename}"
        
        send_url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
        template = {
            "object_type": "feed",
            "content": {
                "title": f"🚨 {event.anomaly_type} 의심 상황 발생!",
                "description": f"시간: {event.timestamp}\n사건ID: {event.event_id}",
                "image_url": image_public_url,
                "image_width": 640,
                "image_height": 480,
                "link": {
                    "web_url": public_domain,
                    "mobile_web_url": public_domain
                }
            },
            "buttons": [
                {
                    "title": "대시보드 확인",
                    "link": {"web_url": public_domain, "mobile_web_url": public_domain}
                }
            ]
        }
        
        data = {"template_object": json.dumps(template)}
        res = requests.post(send_url, headers=headers, data=data)
        
        if res.status_code == 200:
            print("💬 카카오톡 메시지(사진 포함) 전송 성공!")
        elif res.status_code == 401 and not is_retry:
            # 401 Unauthorized: 토큰 만료됨. 갱신 후 1회 재시도
            print("⚠️ 액세스 토큰이 만료되었습니다. 갱신을 시도합니다...")
            if refresh_kakao_token():
                send_kakao_message(event, is_retry=True)
        else:
            print(f"❌ 카카오톡 전송 실패: [{res.status_code}] {res.text}")
            
    except Exception as e:
        print(f"❌ 카카오톡 API 호출 중 오류 발생: {e}")

# 카카오 키 받는 용도 
@app.get("/kakao/login")
def kakao_login():
    """최초 1회 로그인을 위한 엔드포인트"""
    url = f"https://kauth.kakao.com/oauth/authorize?client_id={KAKAO_REST_API_KEY}&redirect_uri={KAKAO_REDIRECT_URI}&response_type=code"

    return RedirectResponse(url)

@app.get("/kakao/callback")
def kakao_callback(code: str):
    """카카오 로그인 완료 후 토큰을 받아오는 콜백 엔드포인트"""
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": KAKAO_REST_API_KEY,
        "redirect_uri": KAKAO_REDIRECT_URI,
        "code": code
    }
    res = requests.post(url, data=data)
    if res.status_code == 200:
        tokens = res.json()
        access_token = tokens.get("access_token")
        user_id = None
        if access_token:
            me_res = requests.get(
                "https://kapi.kakao.com/v2/user/me",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            if me_res.status_code == 200:
                user_info = me_res.json()
                user_id = user_info.get("id")

        if user_id is not None:
            save_user_tokens(user_id, tokens)
            return HTMLResponse(
                f"<h2>✅ 카카오 로그인이 완료되었습니다!</h2>"
                f"<p>사용자 ID: {user_id}로 알림 전송이 설정되었습니다.</p>"
                "<p>이제 이 창을 닫고 프로그램을 계속 사용하셔도 됩니다.</p>"
            )

        save_user_tokens("default", tokens)
        return HTMLResponse(
            "<h2>✅ 카카오 로그인이 완료되었습니다!</h2>"
            "<p>사용자 정보를 가져올 수 없어 기본 알림 계정으로 저장되었습니다.</p>"
            "<p>이제 이 창을 닫고 프로그램을 계속 사용하셔도 됩니다.</p>"
        )
    return {"error": "로그인 실패", "details": res.text}

@app.post("/api/alert")
def receive_alert(event: AnomalyEvent):
    # AI가 없어서 데이터가 잘들어오는지 터미널에서만 출력 
    print("=" * 40)
    print(f" *** 이상행동 감지 알림 수신! *** ")
    print(f" - 사건ID: {event.event_id}")
    print(f" - 발생시간: {event.timestamp}")
    print(f" - 행동유형: {event.anomaly_type}")
    print(f" - 사진경로: {event.image_path}")
    print(f" - 신뢰도: {event.anomaly_score * 100:.1f}%")
    print("=" * 40)

    if event.anomaly_score < ALERT_CONFIDENCE_THRESHOLD:
        print(f"⚠️ 알림 생략: 이상행동 신뢰도 {event.anomaly_score * 100:.1f}% < {ALERT_CONFIDENCE_THRESHOLD * 100:.0f}%")
        return {
            "status": "ignored",
            "message": "알림 신뢰도가 낮아 전송되지 않았습니다.",
            "confidence": event.anomaly_score
        }

    # 새로 들어온 알림을 리스트에 추가 (웹에서 보이도록)
    alerts_db.append(event)

    # 카카오톡 알림 전송 로직 실행 (분리된 함수 호출)
    send_kakao_message(event)

    # 요청을 보낸 쪽(CCTV 파트)에 잘받았다고 응답
    return {"status": "success", "message": "알림 데이터 수신 완료"}