<CourseFloatingBanner
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/#fileId=https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/code_agents.ipynb"},
]}
askForHelpUrl="http://hf.co/join/discord"
 />

# 코드를 사용하는 에이전트 구축하기

코드 에이전트는 `smolagents`의 기본 에이전트 유형입니다. 이들은 작업을 수행하기 위해 Python 도구 호출을 생성하여 효율적이고 표현력이 풍부하며 정확한 작업 표현을 달성합니다.

이들의 간소화된 접근 방식은 필요한 작업 수를 줄이고, 복잡한 작업을 단순화하며, 기존 코드 함수의 재사용을 가능하게 합니다. `smolagents`는 약 1,000줄의 코드로 구현된 코드 에이전트 구축을 위한 경량 프레임워크를 제공합니다.

![Code vs JSON Actions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png)
[실행 가능한 코드 액션이 더 나은 LLM 에이전트를 유도한다](https://huggingface.co/papers/2402.01030) 논문의 그래픽

<Tip>
코드 에이전트가 왜 효과적인지에 대해 더 자세히 알고 싶다면, smolagents 문서의 <a href="https://huggingface.co/docs/smolagents/en/conceptual_guides/intro_agents#code-agents" target="_blank">이 가이드</a>를 확인해 보세요.
</Tip>

## 왜 코드 에이전트인가?

다단계 에이전트 프로세스에서 LLM은 작업을 작성하고 실행하며, 일반적으로 외부 도구 호출을 포함합니다. 전통적인 접근 방식은 도구 이름과 인수를 문자열로 지정하기 위해 JSON 형식을 사용하며, **시스템은 어떤 도구를 실행할지 결정하기 위해 이를 구문 분석해야 합니다**.

그러나 연구에 따르면 **도구 호출 LLM은 코드를 직접 사용하는 것이 더 효과적**입니다. 이는 위에 표시된 [실행 가능한 코드 액션이 더 나은 LLM 에이전트를 유도한다](https://huggingface.co/papers/2402.01030) 논문의 다이어그램에서 볼 수 있듯이 `smolagents`의 핵심 원칙입니다.

JSON 대신 코드로 작업을 작성하는 것은 다음과 같은 몇 가지 주요 이점을 제공합니다:

*   **구성 가능성**: 작업을 쉽게 결합하고 재사용할 수 있습니다.
*   **객체 관리**: 이미지와 같은 복잡한 구조를 직접 다룰 수 있습니다.
*   **일반성**: 계산적으로 가능한 모든 작업을 표현할 수 있습니다.
*   **LLM에 자연스러움**: 고품질 코드는 이미 LLM 훈련 데이터에 존재합니다.

## 코드 에이전트는 어떻게 작동하는가?

![From https://huggingface.co/docs/smolagents/conceptual_guides/react](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/codeagent_docs.png)

위 다이어그램은 Unit 1에서 언급했던 ReAct 프레임워크를 따라 `CodeAgent.run()`이 어떻게 작동하는지 보여줍니다. `smolagents`에서 에이전트의 주요 추상화는 핵심 빌딩 블록 역할을 하는 `MultiStepAgent`입니다. `CodeAgent`는 아래 예시에서 볼 수 있듯이 `MultiStepAgent`의 특별한 종류입니다.

`CodeAgent`는 기존 변수와 지식이 에이전트의 컨텍스트(실행 로그에 보관됨)에 통합되는 일련의 단계를 통해 작업을 수행합니다:

1.  시스템 프롬프트는 `SystemPromptStep`에 저장되고, 사용자 쿼리는 `TaskStep`에 기록됩니다.

2.  그런 다음, 다음 while 루프가 실행됩니다:

    2.1 `agent.write_memory_to_messages()` 메서드는 에이전트의 로그를 LLM이 읽을 수 있는 [채팅 메시지](https://huggingface.co/docs/transformers/main/en/chat_templating) 목록으로 작성합니다.

    2.2 이 메시지들은 `Model`로 전송되어 완성을 생성합니다.

    2.3 완성된 내용은 액션을 추출하기 위해 구문 분석되며, 이 경우 `CodeAgent`를 사용하므로 코드 스니펫이어야 합니다.

    2.4 액션이 실행됩니다.

    2.5 결과는 `ActionStep`에 메모리에 기록됩니다.

각 단계가 끝날 때, 에이전트에 함수 호출(`agent.step_callback`에)이 포함되어 있으면 실행됩니다.

## 몇 가지 예시를 살펴보자

<Tip>
Google Colab을 사용하여 실행할 수 있는 <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/code_agents.ipynb" target="_blank">이 노트북</a>의 코드를 따라할 수 있습니다.
</Tip>

알프레드는 웨인 가문의 저택에서 파티를 계획 중이며 모든 일이 순조롭게 진행되도록 당신의 도움이 필요합니다. 그를 돕기 위해, 우리는 다단계 `CodeAgent`가 어떻게 작동하는지에 대해 배운 것을 적용할 것입니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/smolagents/alfred-party.jpg" alt="Alfred Party"/>

아직 `smolagents`를 설치하지 않았다면 다음 명령을 실행하여 설치할 수 있습니다:

```bash
pip install smolagents -U
```

또한 Hugging Face Hub에 로그인하여 Serverless Inference API에 액세스할 수 있도록 합시다.

```python
from huggingface_hub import login

login()
```

### `smolagents`를 사용하여 파티 플레이리스트 선택하기

음악은 성공적인 파티의 필수적인 부분입니다! 알프레드는 플레이리스트를 선택하는 데 도움이 필요합니다. 다행히 `smolagents`가 우리를 도와줄 수 있습니다! DuckDuckGo를 사용하여 웹을 검색할 수 있는 에이전트를 구축할 수 있습니다. 에이전트가 이 도구에 액세스할 수 있도록 에이전트를 생성할 때 도구 목록에 포함합니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/smolagents/alfred-playlist.jpg" alt="Alfred Playlist"/>

모델의 경우, Hugging Face의 [Serverless Inference API](https://huggingface.co/docs/api-inference/index)에 액세스를 제공하는 `InferenceClientModel`에 의존할 것입니다. 기본 모델은 `"Qwen/Qwen2.5-Coder-32B-Instruct"`이며, 이는 성능이 좋고 빠른 추론에 사용할 수 있지만, Hub에서 호환되는 모든 모델을 선택할 수 있습니다.

에이전트를 실행하는 것은 매우 간단합니다:

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=InferenceClientModel())

agent.run("웨인 저택 파티를 위한 최고의 음악 추천을 검색해 줘.")
```

이 예시를 실행하면 출력은 **실행되는 워크플로우 단계의 추적을 표시**합니다. 또한 다음 메시지와 함께 해당 Python 코드를 인쇄합니다:

```python
 ─ Executing parsed code: ────────────────────────────────────────────────────────────────────────────────────────
  results = web_search(query="best music for a Batman party")
  print(results)
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

몇 단계 후에 알프레드가 파티에 사용할 수 있는 생성된 플레이리스트를 볼 수 있습니다! 🎵

### 사용자 정의 도구를 사용하여 메뉴 준비하기

<img src="https://huggingface.co/datasets/huggingface/course-images/resolve/main/en/unit2/smolagents/alfred-menu.jpg" alt="Alfred Menu"/>

이제 플레이리스트를 선택했으니, 손님들을 위한 메뉴를 정리해야 합니다. 다시 한번, 알프레드는 `smolagents`를 활용할 수 있습니다. 여기서는 `@tool` 데코레이터를 사용하여 도구 역할을 하는 사용자 정의 함수를 정의합니다. 도구 생성에 대해서는 나중에 더 자세히 다룰 것이므로, 지금은 단순히 코드를 실행할 수 있습니다.

아래 예시에서 볼 수 있듯이, `@tool` 데코레이터를 사용하여 도구를 생성하고 `tools` 목록에 포함할 것입니다.

```python
from smolagents import CodeAgent, tool, InferenceClientModel

# 행사에 따라 메뉴를 제안하는 도구
@tool
def suggest_menu(occasion: str) -> str:
    """
    행사에 따라 메뉴를 제안합니다.
    Args:
        occasion (str): 파티의 종류. 허용되는 값은 다음과 같습니다:
                        - "casual": 캐주얼 파티 메뉴.
                        - "formal": 격식 있는 파티 메뉴.
                        - "superhero": 슈퍼히어로 파티 메뉴.
                        - "custom": 사용자 정의 메뉴.
    """
    if occasion == "casual":
        return "피자, 스낵, 음료."
    elif occasion == "formal":
        return "와인과 디저트가 포함된 3코스 저녁 식사."
    elif occasion == "superhero":
        return "고에너지 및 건강식 뷔페."
    else:
        return "집사를 위한 맞춤 메뉴."

# 집사 알프레드가 파티 메뉴를 준비합니다.
agent = CodeAgent(tools=[suggest_menu], model=InferenceClientModel())

# 파티 메뉴 준비
agent.run("파티를 위한 격식 있는 메뉴를 준비해 줘.")
```

에이전트는 답을 찾을 때까지 몇 단계를 실행할 것입니다. 독스트링에 허용되는 값을 명시하면 에이전트가 `occasion` 인수의 기존 값으로 향하고 환각을 제한하는 데 도움이 됩니다.

메뉴가 준비되었습니다! 🥗

### 에이전트 내에서 Python 임포트 사용하기

플레이리스트와 메뉴는 준비되었지만, 한 가지 중요한 세부 사항인 준비 시간을 확인해야 합니다!

알프레드는 다른 슈퍼히어로들의 도움이 필요할 경우, 지금 준비를 시작하면 모든 것이 언제 준비될지 계산해야 합니다.

`smolagents`는 Python 코드 스니펫을 작성하고 실행하는 에이전트를 전문으로 하며, 보안을 위해 샌드박스 실행을 제공합니다.

**코드 실행에는 엄격한 보안 조치가 있습니다** - 미리 정의된 안전 목록 외의 임포트는 기본적으로 차단됩니다. 그러나 `additional_authorized_imports`에 문자열로 전달하여 추가 임포트를 승인할 수 있습니다.
보안 코드 실행에 대한 자세한 내용은 공식 [가이드](https://huggingface.co/docs/smolagents/tutorials/secure_code_execution)를 참조하십시오.

에이전트를 생성할 때 `additional_authorized_imports`를 사용하여 `datetime` 모듈을 임포트할 수 있도록 허용할 것입니다.

```python
from smolagents import CodeAgent, InferenceClientModel
import numpy as np
import time
import datetime

agent = CodeAgent(tools=[], model=InferenceClientModel(), additional_authorized_imports=['datetime'])

agent.run(
    """
    알프레드는 파티를 준비해야 합니다. 다음은 작업 목록입니다:
    1. 음료 준비 - 30분
    2. 저택 장식 - 60분
    3. 메뉴 설정 - 45분
    4. 음악 및 플레이리스트 준비 - 45분

    지금 바로 시작한다면, 파티는 몇 시에 준비될까요?
    """
)
```


이 예시들은 코드 에이전트로 할 수 있는 일의 시작에 불과하며, 우리는 이미 파티 준비에 대한 유용성을 보기 시작했습니다.
코드 에이전트를 구축하는 방법에 대한 자세한 내용은 [smolagents 문서](https://huggingface.co/docs/smolagents)에서 확인할 수 있습니다.

요약하자면, `smolagents`는 Python 코드 스니펫을 작성하고 실행하는 에이전트를 전문으로 하며, 보안을 위해 샌드박스 실행을 제공합니다. 로컬 및 API 기반 언어 모델을 모두 지원하여 다양한 개발 환경에 적응할 수 있습니다.

### 커스텀 파티 준비 에이전트를 허브에 공유하기

**우리만의 알프레드 에이전트를 커뮤니티와 공유하는 것이 멋지지 않을까요**? 그렇게 함으로써, 누구나 허브에서 직접 에이전트를 쉽게 다운로드하고 사용할 수 있으며, 고담의 궁극적인 파티 플래너를 손끝으로 가져올 수 있습니다! 실현시켜 봅시다! 🎉

`smolagents` 라이브러리는 완전한 에이전트를 커뮤니티와 공유하고 다른 에이전트를 즉시 사용할 수 있도록 다운로드할 수 있게 함으로써 이를 가능하게 합니다. 다음처럼 간단합니다:

```python
# 사용자 이름과 저장소 이름으로 변경
agent.push_to_hub('sergiopaniego/AlfredAgent')
```

에이전트를 다시 다운로드하려면 아래 코드를 사용하십시오:

```python
# 사용자 이름과 저장소 이름으로 변경
alfred_agent = agent.from_hub('sergiopaniego/AlfredAgent', trust_remote_code=True)

alfred_agent.run("웨인 저택 파티를 위한 최고의 플레이리스트를 알려줘. 파티 아이디어는 '악당 가면무도회' 테마야.")
```

또한 흥미로운 점은 공유된 에이전트가 Hugging Face Spaces로 직접 제공되어 실시간으로 상호 작용할 수 있다는 것입니다. 다른 에이전트는 [여기](https://huggingface.co/spaces/davidberenstein1957/smolagents-and-tools)에서 탐색할 수 있습니다.

예를 들어, _AlfredAgent_는 [여기](https://huggingface.co/spaces/sergiopaniego/AlfredAgent)에서 사용할 수 있습니다. 아래에서 직접 사용해 볼 수 있습니다:

<iframe
	src="https://sergiopaniego-alfredagent.hf.space/"
	frameborder="0"
	width="850"
	height="450"
></iframe>

알프레드가 `smolagents`를 사용하여 어떻게 이런 에이전트를 구축했는지 궁금할 것입니다. 여러 도구를 통합하여 다음과 같이 에이전트를 생성할 수 있습니다. 지금은 도구에 대해 걱정하지 마십시오. 이 유닛의 뒷부분에서 자세히 다룰 전용 섹션이 있습니다:

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, Tool, tool, VisitWebpageTool

@tool
def suggest_menu(occasion: str) -> str:
    """
    행사에 따라 메뉴를 제안합니다.
    Args:
        occasion: 파티의 종류.
    """
    if occasion == "casual":
        return "피자, 스낵, 음료."
    elif occasion == "formal":
        return "와인과 디저트가 포함된 3코스 저녁 식사."
    elif occasion == "superhero":
        return "고에너지 및 건강식 뷔페."
    else:
        return "집사를 위한 맞춤 메뉴."

@tool
def catering_service_tool(query: str) -> str:
    """
    이 도구는 고담 시에서 가장 높은 평가를 받은 케이터링 서비스를 반환합니다.

    Args:
        query: 케이터링 서비스를 찾기 위한 검색어.
    """
    # 케이터링 서비스 및 평점 예시 목록
    services = {
        "고담 케이터링 Co.": 4.9,
        "웨인 매너 케이터링": 4.8,
        "고담 시티 이벤트": 4.7,
    }

    # 가장 높은 평가를 받은 케이터링 서비스 찾기 (검색어 필터링 시뮬레이션)
    best_service = max(services, key=services.get)

    return best_service

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    이 도구는 카테고리를 기반으로 창의적인 슈퍼히어로 테마 파티 아이디어를 제안합니다.
    고유한 파티 테마 아이디어를 반환합니다."""

    inputs = {
        "category": {
            "type": "string",
            "description": "슈퍼히어로 파티의 종류 (예: '클래식 영웅', '악당 가면무도회', '미래 고담').",
        }
    }

    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "저스티스 리그 갈라: 손님들은 좋아하는 DC 영웅으로 분장하고 '크립토나이트 펀치'와 같은 테마 칵테일을 즐깁니다.",
            "villain masquerade": "고담 악당의 무도회: 손님들이 고전 배트맨 악당으로 분장하는 신비로운 가면무도회.",
            "futuristic Gotham": "네오-고담 나이트: 네온 장식과 미래형 장치로 배트맨 비욘드에서 영감을 받은 사이버펑크 스타일 파티."
        }

        return themes.get(category.lower(), "테마 파티 아이디어를 찾을 수 없습니다. '클래식 영웅', '악당 가면무도회' 또는 '미래 고담'을 시도해 보세요.")


# 집사 알프레드가 파티 메뉴를 준비합니다.
agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        suggest_menu,
        catering_service_tool,
        SuperheroPartyThemeTool(),
	FinalAnswerTool()
    ],
    model=InferenceClientModel(),
    max_steps=10,
    verbosity_level=2
)

agent.run("웨인 저택 파티를 위한 최고의 플레이리스트를 알려줘. 파티 아이디어는 '악당 가면무도회' 테마야.")
```

보시다시피, 우리는 에이전트의 기능을 향상시키는 여러 도구를 가진 `CodeAgent`를 만들었으며, 이를 통해 궁극적인 파티 플래너를 커뮤니티와 공유할 준비가 되었습니다! 🎉

이제 당신의 차례입니다: 방금 배운 지식을 사용하여 자신만의 에이전트를 만들고 커뮤니티와 공유해 보세요! 🕵️‍♂️💡

<Tip>
에이전트 프로젝트를 공유하고 싶다면, 스페이스를 만들고 Hugging Face Hub에서 <a href="https://huggingface.co/agents-course">agents-course</a>를 태그하세요. 여러분이 만든 것을 보고 싶습니다!
</Tip>

### OpenTelemetry 및 Langfuse로 파티 준비 에이전트 검사하기 📡

알프레드가 파티 준비 에이전트를 미세 조정하면서, 실행 디버깅에 지쳐가고 있습니다. 에이전트는 본질적으로 예측 불가능하고 검사하기 어렵습니다. 그러나 그는 궁극적인 파티 준비 에이전트를 구축하고 프로덕션에 배포하는 것을 목표로 하므로, 향후 모니터링 및 분석을 위한 강력한 추적 가능성이 필요합니다.

다시 한번, `smolagents`가 구원하러 왔습니다! 에이전트 실행을 계측하기 위한 [OpenTelemetry](https://opentelemetry.io/) 표준을 채택하여 원활한 검사 및 로깅을 가능하게 합니다. [Langfuse](https://langfuse.com/)와 `SmolagentsInstrumentor`의 도움으로 알프레드는 에이전트의 동작을 쉽게 추적하고 분석할 수 있습니다.

설정은 간단합니다!

먼저, 필요한 종속성을 설치해야 합니다:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents langfuse
```

다음으로, 알프레드는 이미 Langfuse에 계정을 만들었고 API 키를 준비했습니다. 아직 만들지 않았다면 [여기](https://cloud.langfuse.com/)에서 Langfuse Cloud에 가입하거나 [대안](https://huggingface.co/docs/smolagents/tutorials/inspect_runs)을 탐색할 수 있습니다.

API 키를 얻었다면 다음과 같이 올바르게 구성해야 합니다:

```python
import os

# 프로젝트 설정 페이지에서 프로젝트 키를 가져옵니다: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU 지역
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US 지역
```

환경 변수가 설정되었으므로 이제 Langfuse 클라이언트를 초기화할 수 있습니다. get_client()는 환경 변수에 제공된 자격 증명을 사용하여 Langfuse 클라이언트를 초기화합니다.

```python
from langfuse import get_client

langfuse = get_client()

# 연결 확인
if langfuse.auth_check():
    print("Langfuse 클라이언트가 인증되었고 준비되었습니다!")
else:
    print("인증에 실패했습니다. 자격 증명과 호스트를 확인하십시오.")
```

마지막으로, 알프레드는 `SmolagentsInstrumentor`를 초기화하고 에이전트의 성능을 추적할 준비가 되었습니다.

```python
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

SmolagentsInstrumentor().instrument()
```

알프레드가 이제 연결되었습니다 🔌! `smolagents`의 실행이 Langfuse에 기록되어 에이전트의 동작에 대한 완전한 가시성을 제공합니다. 이 설정을 통해 그는 이전 실행을 다시 방문하고 파티 준비 에이전트를 더욱 개선할 준비가 되었습니다.

<Tip>에이전트를 추적하고 수집된 데이터를 사용하여 성능을 평가하는 방법에 대해 더 자세히 알아보려면 <a href="https://huggingface.co/learn/agents-course/bonus-unit2/introduction">보너스 유닛 2</a>를 확인하십시오.</Tip>

```python
from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(tools=[], model=InferenceClientModel())
alfred_agent = agent.from_hub('sergiopaniego/AlfredAgent', trust_remote_code=True)
alfred_agent.run("웨인 저택 파티를 위한 최고의 플레이리스트를 알려줘. 파티 아이디어는 '악당 가면무도회' 테마야.")
```

알프레드는 이제 [여기](https://cloud.langfuse.com/project/cm7bq0abj025rad078ak3luwi/traces/995fc019255528e4f48cf6770b0ce27b?timestamp=2025-02-19T10%3A28%3A36.929Z)에서 이 로그에 액세스하여 검토하고 분석할 수 있습니다.

<Tip>
실제로 실행 중에 사소한 오류가 발생했습니다. 로그에서 찾을 수 있습니까? 에이전트가 이를 어떻게 처리하고 여전히 유효한 답변을 반환하는지 추적해 보세요. <a href="https://cloud.langfuse.com/project/cm7bq0abj025rad078ak3luwi/traces/995fc019255528e4f48cf6770b0ce27b?timestamp=2025-02-19T10%3A28%3A36.929Z&observation=80ca57ace4f69b52">여기</a>는 답변을 확인하고 싶다면 오류에 대한 직접 링크입니다. 물론 오류는 그 사이에 수정되었으며, 자세한 내용은 <a href="https://github.com/huggingface/smolagents/issues/838">이 이슈</a>에서 확인할 수 있습니다.
</Tip>

한편, [제안된 플레이리스트](https://open.spotify.com/playlist/0gZMMHjuxMrrybQ7wTMTpw)는 파티 준비에 완벽한 분위기를 조성합니다. 멋지죠? 🎶

---

이제 첫 번째 코드 에이전트를 만들었으니, `smolagents`에서 사용할 수 있는 두 번째 유형의 에이전트인 **도구 호출 에이전트를 만드는 방법**을 배워봅시다.

## 자료

-   [smolagents 블로그](https://huggingface.co/blog/smolagents) - smolagents 및 코드 상호 작용 소개
-   [smolagents: 좋은 에이전트 구축하기](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - 신뢰할 수 있는 에이전트를 위한 모범 사례
-   [효과적인 에이전트 구축 - Anthropic](https://www.anthropic.com/research/building-effective-agents) - 에이전트 설계 원칙
-   [OpenTelemetry로 실행 공유](https://huggingface.co/docs/smolagents/tutorials/inspect_runs) - 에이전트 추적을 위한 OpenTelemetry 설정 방법에 대한 세부 정보