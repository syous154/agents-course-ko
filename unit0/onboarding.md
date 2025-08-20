# 온보딩: 첫걸음 ⛵

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit0/time-to-onboard.jpg" alt="Time to Onboard" width="100%"/>

이제 모든 세부 사항을 알았으니 시작해 봅시다! 우리는 네 가지를 할 것입니다:

1.  **Hugging Face 계정 생성** (아직 만들지 않았다면)
2.  **Discord에 가입하고 자신을 소개하기** (부끄러워하지 마세요 🤗)
3.  **Hub에서 Hugging Face Agents Course 팔로우하기**
4.  **코스에 대해 널리 알리기**

### 1단계: Hugging Face 계정 생성

(아직 만들지 않았다면) [여기](https://huggingface.co/join)에서 Hugging Face 계정을 만드세요.

### 2단계: Discord 커뮤니티 가입

👉🏻 [여기](https://discord.gg/UrrTSsSyjb)에서 Discord 서버에 가입하세요.

가입할 때 `#introduce-yourself` 채널에 자신을 소개하는 것을 잊지 마세요.

우리는 여러 AI 에이전트 관련 채널을 운영하고 있습니다:
-   `agents-course-announcements`: **최신 코스 정보**를 위한 채널입니다.
-   `🎓-agents-course-general`: **일반적인 토론 및 잡담**을 위한 채널입니다.
-   `agents-course-questions`: **질문하고 동료들을 돕기** 위한 채널입니다.
-   `agents-course-showcase`: **최고의 에이전트를 선보이기** 위한 채널입니다.

또한 다음을 확인할 수 있습니다:

-   `smolagents`: **라이브러리에 대한 토론 및 지원**을 위한 채널입니다.

Discord를 처음 사용하신다면, 최상의 활용법을 위해 Discord 101을 작성했습니다. [다음 섹션](discord101)을 확인하세요.

### 3단계: Hugging Face Agent Course Organization 팔로우

**Hugging Face Agents Course Organization을 팔로우하여** 최신 코스 자료, 업데이트 및 공지 사항을 받아보세요.

👉 [여기](https://huggingface.co/agents-course)로 이동하여 **팔로우**를 클릭하세요.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/hf_course_follow.gif" alt="Follow" width="100%"/>

### 4단계: 코스에 대해 널리 알리기

이 코스를 더 널리 알릴 수 있도록 도와주세요! 두 가지 방법으로 도와주실 수 있습니다:

1.  코스 [저장소](https://github.com/huggingface/agents-course)에 ⭐를 눌러 지원을 보여주세요.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/please_star.gif" alt="Repo star"/>

2.  학습 여정 공유: 다른 사람들에게 **이 코스를 수강하고 있다는 것을 알려주세요**! 소셜 미디어 게시물에 사용할 수 있는 일러스트를 준비했습니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/share.png">

이미지를 다운로드하려면 👉 [여기](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/share.png?download=true)를 클릭하세요.

### 5단계: Ollama를 사용하여 로컬에서 모델 실행하기 (크레딧 한도에 도달한 경우)

1.  **Ollama 설치**

    [여기](https://ollama.com/download)에서 공식 지침을 따르세요.

2.  **로컬에서 모델 풀하기**

    ```bash
    ollama pull qwen2:7b
    ```

    여기서는 [qwen2:7b 모델](https://ollama.com/library/qwen2:7b)을 풀합니다. [ollama 웹사이트](https://ollama.com/search)에서 더 많은 모델을 확인하세요.

3.  **백그라운드에서 Ollama 시작하기 (하나의 터미널에서)**
    ``` bash
    ollama serve
    ```

    "listen tcp 127.0.0.1:11434: bind: address already in use" 오류가 발생하면 `sudo lsof -i :11434` 명령을 사용하여 현재 이 포트를 사용 중인 프로세스 ID(PID)를 식별할 수 있습니다. 프로세스가 `ollama`인 경우, 위의 설치 스크립트가 ollama 서비스를 시작했을 가능성이 높으므로 이 명령을 건너뛰고 Ollama를 시작할 수 있습니다.

4.  **`InferenceClientModel` 대신 `LiteLLMModel` 사용하기**

    `smolagents`에서 `LiteLLMModel` 모듈을 사용하려면 `pip` 명령을 실행하여 모듈을 설치해야 합니다.

    ``` bash
    pip install 'smolagents[litellm]'
    ```

    ``` python
    from smolagents import LiteLLMModel

    model = LiteLLMModel(
        model_id="ollama_chat/qwen2:7b",  # 또는 다른 Ollama 지원 모델을 시도하세요.
        api_base="http://127.0.0.1:11434",  # 기본 Ollama 로컬 서버
        num_ctx=8192,
    )
    ```

5.  **이것이 작동하는 이유?**
    -   Ollama는 `http://localhost:11434`에서 OpenAI 호환 API를 사용하여 로컬에서 모델을 제공합니다.
    -   `LiteLLMModel`은 OpenAI 채팅/완성 API 형식을 지원하는 모든 모델과 통신하도록 구축되었습니다.
    -   이는 `InferenceClientModel`을 `LiteLLMModel`로 간단히 교체할 수 있으며 다른 코드 변경이 필요 없다는 것을 의미합니다. 원활하고 플러그 앤 플레이 방식의 솔루션입니다.

축하합니다! 🎉 **온보딩 프로세스를 완료했습니다**! 이제 AI 에이전트에 대해 배울 준비가 되었습니다. 즐거운 시간 보내세요!

계속 배우고, 멋진 모습 유지하세요 🤗