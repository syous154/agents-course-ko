<CourseFloatingBanner 
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/#fileId=https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/tools.ipynb"},
]} 
askForHelpUrl="http://hf.co/join/discord" />

# 도구

[1단원](https://huggingface.co/learn/agents-course/unit1/tools)에서 살펴본 바와 같이 에이전트는 다양한 작업을 수행하기 위해 도구를 사용합니다. `smolagents`에서 도구는 **LLM이 에이전트 시스템 내에서 호출할 수 있는 함수**로 처리됩니다.

도구와 상호 작용하려면 LLM에 다음과 같은 주요 구성 요소가 포함된 **인터페이스 설명**이 필요합니다.

- **이름**: 도구의 이름
- **도구 설명**: 도구가 수행하는 작업
- **입력 유형 및 설명**: 도구가 허용하는 인수
- **출력 유형**: 도구가 반환하는 내용

예를 들어, 웨인 저택에서 파티를 준비하는 동안 알프레드는 케이터링 서비스 검색에서 파티 테마 아이디어 찾기에 이르기까지 정보를 수집하기 위해 다양한 도구가 필요합니다. 간단한 검색 도구 인터페이스는 다음과 같습니다.

- **이름:** `web_search`
- **도구 설명:** 특정 쿼리에 대해 웹을 검색합니다.
- **입력:** `query`(문자열) - 조회할 검색어
- **출력:** 검색 결과가 포함된 문자열

이러한 도구를 사용하여 알프레드는 정보에 입각한 결정을 내리고 완벽한 파티를 계획하는 데 필요한 모든 정보를 수집할 수 있습니다.

아래에서 도구 호출이 관리되는 방식을 보여주는 애니메이션을 볼 수 있습니다.

![https://huggingface.co/docs/smolagents/conceptual_guides/react의 에이전트 파이프라인](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif)

## 도구 생성 방법

`smolagents`에서 도구는 두 가지 방법으로 정의할 수 있습니다.
1. 간단한 함수 기반 도구에 **`@tool` 데코레이터 사용**
2. 더 복잡한 기능을 위해 **`Tool`의 하위 클래스 생성**

### `@tool` 데코레이터

`@tool` 데코레이터는 **간단한 도구를 정의하는 권장 방법**입니다. 내부적으로 smolagents는 Python에서 함수에 대한 기본 정보를 구문 분석합니다. 따라서 함수 이름을 명확하게 지정하고 좋은 독스트링을 작성하면 LLM이 더 쉽게 사용할 수 있습니다.

이 접근 방식을 사용하여 다음을 사용하여 함수를 정의합니다.

- **LLM이 목적을 이해하는 데 도움이 되는 명확하고 설명적인 함수 이름**.
- **적절한 사용을 보장하기 위한 입력 및 출력에 대한 유형 힌트**.
- **각 인수가 명시적으로 설명된 `Args:` 섹션을 포함한 자세한 설명**. 이러한 설명은 LLM에 귀중한 컨텍스트를 제공하므로 신중하게 작성하는 것이 중요합니다.

#### 최고 등급 케이터링을 검색하는 도구 생성

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/smolagents/alfred-catering.jpg" alt="알프레드 케이터링"/>

<Tip>
Google Colab을 사용하여 실행할 수 있는 <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/tools.ipynb" target="_blank">이 노트북</a>의 코드를 따를 수 있습니다.
</Tip>

알프레드가 이미 파티 메뉴를 결정했지만 이제 많은 손님을 위해 음식을 준비하는 데 도움이 필요하다고 상상해 봅시다. 이를 위해 그는 케이터링 서비스를 고용하고 싶어하며 사용 가능한 최고 등급 옵션을 식별해야 합니다. 알프레드는 도구를 활용하여 해당 지역 최고의 케이터링 서비스를 검색할 수 있습니다.

다음은 알프레드가 `@tool` 데코레이터를 사용하여 이를 수행하는 방법의 예입니다.

```python
from smolagents import CodeAgent, InferenceClientModel, tool

# 최고 등급 케이터링 서비스를 가져오는 함수가 있다고 가정해 보겠습니다.
@tool
def catering_service_tool(query: str) -> str:
    """
    이 도구는 고담시에서 가장 높은 등급의 케이터링 서비스를 반환합니다.

    Args:
        query: 케이터링 서비스를 찾기 위한 검색어입니다.
    """
    # 케이터링 서비스 및 등급의 예시 목록
    services = {
        "고담 케이터링 주식회사": 4.9,
        "웨인 저택 케이터링": 4.8,
        "고담시 이벤트": 4.7,
    }

    # 가장 높은 등급의 케이터링 서비스 찾기(검색 쿼리 필터링 시뮬레이션)
    best_service = max(services, key=services.get)

    return best_service


agent = CodeAgent(tools=[catering_service_tool], model=InferenceClientModel())

# 최고의 케이터링 서비스를 찾기 위해 에이전트 실행
result = agent.run(
    "고담시에서 가장 높은 등급의 케이터링 서비스 이름을 알려주시겠어요?"
)

print(result) # 출력: 고담 케이터링 주식회사
```

### 도구를 Python 클래스로 정의

이 접근 방식은 [`Tool`](https://huggingface.co/docs/smolagents/v1.8.1/en/reference/tools#smolagents.Tool)의 하위 클래스를 만드는 것을 포함합니다. 복잡한 도구의 경우 Python 함수 대신 클래스를 구현할 수 있습니다. 클래스는 LLM이 효과적으로 사용하는 방법을 이해하는 데 도움이 되는 메타데이터로 함수를 래핑합니다. 이 클래스에서 우리는 다음을 정의합니다.

- `name`: 도구의 이름.
- `description`: 에이전트의 시스템 프롬프트를 채우는 데 사용되는 설명.
- `inputs`: Python 인터프리터가 입력을 처리하는 데 도움이 되는 정보를 제공하는 `type` 및 `description` 키가 있는 사전.
- `output_type`: 예상 출력 유형을 지정합니다.
- `forward`: 실행할 추론 논리가 포함된 메서드.

아래에서 `Tool`을 사용하여 빌드된 도구의 예와 `CodeAgent` 내에서 통합하는 방법을 볼 수 있습니다.

#### 슈퍼히어로 테마 파티에 대한 아이디어를 생성하는 도구 생성

저택에서 열리는 알프레드의 파티는 **슈퍼히어로 테마 이벤트**이지만, 정말 특별하게 만들기 위해서는 창의적인 아이디어가 필요합니다. 환상적인 호스트로서 그는 독특한 테마로 손님들을 놀라게 하고 싶어합니다.

이를 위해 그는 주어진 카테고리를 기반으로 슈퍼히어로 테마 파티 아이디어를 생성하는 에이전트를 사용할 수 있습니다. 이렇게 하면 알프레드는 손님들을 놀라게 할 완벽한 파티 테마를 찾을 수 있습니다.

```python
from smolagents import Tool, CodeAgent, InferenceClientModel

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    이 도구는 카테고리를 기반으로 창의적인 슈퍼히어로 테마 파티 아이디어를 제안합니다.
    독특한 파티 테마 아이디어를 반환합니다."""

    inputs = {
        "category": {
            "type": "string",
            "description": "슈퍼히어로 파티 유형(예: '클래식 영웅', '악당 가장 무도회', '미래 고담').",
        }
    }

    output_type = "string"

    def forward(self, category: str):
        themes = {
            "클래식 영웅": "저스티스 리그 갈라: 손님들은 '크립토나이트 펀치'와 같은 테마 칵테일과 함께 좋아하는 DC 영웅으로 분장합니다.",
            "악당 가장 무도회": "고담 악당들의 무도회: 손님들이 고전 배트맨 악당으로 분장하는 신비한 가장 무도회.",
            "미래 고담": "네오-고담의 밤: 배트맨 비욘드에서 영감을 받은 사이버펑크 스타일의 파티로, 네온 장식과 미래형 가젯이 있습니다."
        }

        return themes.get(category.lower(), "테마 파티 아이디어를 찾을 수 없습니다. '클래식 영웅', '악당 가장 무도회' 또는 '미래 고담'을 시도해 보세요.")

# 도구 인스턴스화
party_theme_tool = SuperheroPartyThemeTool()
agent = CodeAgent(tools=[party_theme_tool], model=InferenceClientModel())

# 파티 테마 아이디어를 생성하기 위해 에이전트 실행
result = agent.run(
    "'악당 가장 무도회' 테마에 좋은 슈퍼히어로 파티 아이디어는 무엇일까요?"
)

print(result) # 출력: "고담 악당들의 무도회: 손님들이 고전 배트맨 악당으로 분장하는 신비한 가장 무도회."
```

이 도구를 사용하면 알프레드는 최고의 슈퍼 호스트가 되어 잊지 못할 슈퍼히어로 테마 파티로 손님들에게 깊은 인상을 남길 것입니다! 🦸‍♂️🦸‍♀️

## 기본 도구 상자

`smolagents`에는 에이전트에 직접 주입할 수 있는 사전 빌드된 도구 세트가 함께 제공됩니다. [기본 도구 상자](https://huggingface.co/docs/smolagents/guided_tour?build-a-tool=Decorate+a+function+with+%40tool#default-toolbox)에는 다음이 포함됩니다.

- **PythonInterpreterTool**
- **FinalAnswerTool**
- **UserInputTool**
- **DuckDuckGoSearchTool**
- **GoogleSearchTool**
- **VisitWebpageTool**

알프레드는 웨인 저택에서 완벽한 파티를 보장하기 위해 다양한 도구를 사용할 수 있습니다.

- 먼저, 그는 창의적인 슈퍼히어로 테마 파티 아이디어를 찾기 위해 `DuckDuckGoSearchTool`을 사용할 수 있습니다.

- 케이터링의 경우, 그는 고담에서 가장 높은 등급의 서비스를 찾기 위해 `GoogleSearchTool`에 의존할 것입니다.

- 좌석 배치를 관리하기 위해 알프레드는 `PythonInterpreterTool`로 계산을 실행할 수 있습니다.

- 모든 것이 수집되면 그는 `FinalAnswerTool`을 사용하여 계획을 컴파일할 것입니다.

이러한 도구를 통해 알프레드는 파티가 예외적이고 원활하게 진행되도록 보장합니다. 🦇💡

## 도구 공유 및 가져오기

**smolagents**의 가장 강력한 기능 중 하나는 Hub에서 사용자 지정 도구를 공유하고 커뮤니티에서 만든 도구를 원활하게 통합하는 기능입니다. 여기에는 **HF Spaces** 및 **LangChain 도구**와의 연결이 포함되어 알프레드가 웨인 저택에서 잊을 수 없는 파티를 조율하는 능력을 크게 향상시킵니다. 🎭

이러한 통합을 통해 알프레드는 완벽한 분위기를 위한 조명 조정, 파티에 이상적인 재생 목록 선별 또는 고담 최고의 케이터링 업체와의 조정 등 고급 이벤트 계획 도구를 활용할 수 있습니다.

다음은 이러한 기능이 파티 경험을 어떻게 향상시킬 수 있는지 보여주는 예입니다.

### Hub에 도구 공유

사용자 지정 도구를 커뮤니티와 공유하는 것은 쉽습니다! `push_to_hub()` 메서드를 사용하여 Hugging Face 계정에 업로드하기만 하면 됩니다.

예를 들어, 알프레드는 다른 사람들이 고담에서 최고의 케이터링 서비스를 찾는 데 도움이 되도록 `party_theme_tool`을 공유할 수 있습니다. 방법은 다음과 같습니다.

```python
party_theme_tool.push_to_hub("{your_username}/party_theme_tool", token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")
```

### Hub에서 도구 가져오기

`load_tool()` 함수를 사용하여 다른 사용자가 만든 도구를 쉽게 가져올 수 있습니다. 예를 들어, 알프레드는 AI를 사용하여 파티 홍보 이미지를 생성하고 싶을 수 있습니다. 처음부터 도구를 만드는 대신 커뮤니티의 미리 정의된 도구를 활용할 수 있습니다.

```python
from smolagents import load_tool, CodeAgent, InferenceClientModel

image_generation_tool = load_tool(
    "m-ric/text-to-image",
    trust_remote_code=True
)

agent = CodeAgent(
    tools=[image_generation_tool],
    model=InferenceClientModel()
)

agent.run("웨인 저택에서 가상의 슈퍼히어로들과 함께하는 호화로운 슈퍼히어로 테마 파티 이미지를 생성하세요.")
```

### Hugging Face Space를 도구로 가져오기

`Tool.from_space()`를 사용하여 HF Space를 도구로 가져올 수도 있습니다. 이는 이미지 생성에서 데이터 분석에 이르기까지 커뮤니티의 수천 개의 공간과 통합할 수 있는 가능성을 열어줍니다.

이 도구는 `gradio_client`를 사용하여 공간 Gradio 백엔드와 연결되므로 아직 설치하지 않은 경우 `pip`를 통해 설치해야 합니다.

파티를 위해 알프레드는 발표에 사용할 AI 생성 이미지 생성을 위해 기존 HF Space를 사용할 수 있습니다(앞서 언급한 사전 빌드된 도구 대신). 만들어 봅시다!

```python
from smolagents import CodeAgent, InferenceClientModel, Tool

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="프롬프트에서 이미지 생성"
)

model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")

agent = CodeAgent(tools=[image_generation_tool], model=model)

agent.run(
    "이 프롬프트를 개선한 다음 이미지를 생성하세요.",
    additional_args={'user_prompt': '웨인 저택에서 열리는 성대한 슈퍼히어로 테마 파티, 알프레드가 호화로운 갈라를 감독합니다.'}
)
```

### LangChain 도구 가져오기


다가오는 섹션에서 `LangChain` 프레임워크에 대해 논의할 것입니다. 지금은 smolagents 워크플로에서 LangChain 도구를 재사용할 수 있다는 점만 참고하십시오!

`Tool.from_langchain()` 메서드를 사용하여 LangChain 도구를 쉽게 로드할 수 있습니다. 완벽주의자인 알프레드는 웨인 부부가 없는 동안 웨인 저택에서 화려한 슈퍼히어로의 밤을 준비하고 있습니다. 모든 세부 사항이 기대를 뛰어넘도록 하기 위해 그는 LangChain 도구를 활용하여 최고 수준의 엔터테인먼트 아이디어를 찾습니다.

`Tool.from_langchain()`을 사용하여 알프레드는 smolagent에 고급 검색 기능을 손쉽게 추가하여 몇 가지 명령만으로 독점적인 파티 아이디어와 서비스를 발견할 수 있습니다.

방법은 다음과 같습니다.

```python
from langchain.agents import load_tools
from smolagents import CodeAgent, InferenceClientModel, Tool

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools=[search_tool], model=model)

agent.run("라이브 공연 및 인터랙티브 경험과 같은 슈퍼히어로 테마 이벤트에 대한 고급 엔터테인먼트 아이디어를 검색하세요.")
```

### 모든 MCP 서버에서 도구 모음 가져오기

`smolagents`는 또한 [glama.ai](https://glama.ai/mcp/servers) 또는 [smithery.ai](https://smithery.ai)에서 사용할 수 있는 수백 개의 MCP 서버에서 도구를 가져올 수 있습니다. MCP에 대해 더 자세히 알고 싶다면 [무료 MCP 과정](https://huggingface.co/learn/mcp-course/)을 확인하십시오.

<details>
<summary>mcp 클라이언트 설치</summary>

먼저 `smolagents`에 대한 `mcp` 통합을 설치해야 합니다.

```bash
pip install "smolagents[mcp]"
```
</details>

MCP 서버 도구는 다음과 같이 ToolCollection 개체에 로드할 수 있습니다.

```python
import os
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters
from smolagents import InferenceClientModel


model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")


server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=model, add_base_tools=True)
    agent.run("숙취 해소제를 찾아주세요.")
```

이 설정을 통해 알프레드는 고급 엔터테인먼트 옵션을 신속하게 발견하여 고담의 엘리트 손님들이 잊을 수 없는 경험을 할 수 있도록 보장합니다. 이 도구는 웨인 저택에 완벽한 슈퍼히어로 테마 이벤트를 기획하는 데 도움이 됩니다! 🎉

## 리소스

- [도구 튜토리얼](https://huggingface.co/docs/smolagents/tutorials/tools) - 이 튜토리얼을 탐색하여 도구를 효과적으로 사용하는 방법을 배우십시오.
- [도구 설명서](https://huggingface.co/docs/smolagents/v1.8.1/en/reference/tools) - 도구에 대한 포괄적인 참조 설명서.
- [도구 가이드 투어](https://huggingface.co/docs/smolagents/v1.8.1/en/guided_tour#tools) - 도구를 효율적으로 구축하고 활용하는 데 도움이 되는 단계별 가이드 투어.
- [효과적인 에이전트 구축](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - 안정적이고 고성능의 사용자 지정 기능 에이전트를 개발하기 위한 모범 사례에 대한 자세한 가이드. 