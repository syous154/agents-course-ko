다음 텍스트를 한국어로 번역합니다:

<CourseFloatingBanner
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/#fileId=https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/tool_calling_agents.ipynb"},
]}
askForHelpUrl="http://hf.co/join/discord" />

# 코드 스니펫 또는 JSON 블롭으로 액션 작성하기

<Tip>
Google Colab을 사용하여 실행할 수 있는 <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/tool_calling_agents.ipynb" target="_blank">이 노트북</a>의 코드를 따라할 수 있습니다.
</Tip>

도구 호출 에이전트(Tool Calling Agents)는 `smolagents`에서 사용할 수 있는 두 번째 유형의 에이전트입니다. 파이썬 스니펫을 사용하는 코드 에이전트(Code Agents)와 달리, 이 에이전트들은 **LLM 제공업체의 내장 도구 호출 기능**을 사용하여 도구 호출을 **JSON 구조**로 생성합니다. 이는 OpenAI, Anthropic 및 기타 여러 제공업체에서 사용하는 표준 접근 방식입니다.

예시를 살펴보겠습니다. 알프레드가 케이터링 서비스와 파티 아이디어를 검색하고 싶을 때, `CodeAgent`는 다음과 같은 파이썬 코드를 생성하고 실행합니다.

```python
for query in [
    "Best catering services in Gotham City",
    "Party theme ideas for superheroes"
]:
    print(web_search(f"Search for: {query}"))
```

`ToolCallingAgent`는 대신 다음과 같은 JSON 구조를 생성합니다.

```json
[
    {"name": "web_search", "arguments": "Best catering services in Gotham City"},
    {"name": "web_search", "arguments": "Party theme ideas for superheroes"}
]
```

이 JSON 블롭은 도구 호출을 실행하는 데 사용됩니다.

`smolagents`는 주로 `CodeAgents`에 중점을 두지만([전반적으로 더 나은 성능](https://huggingface.co/papers/2402.01030)을 보이기 때문), `ToolCallingAgents`는 변수 처리나 복잡한 도구 호출이 필요 없는 간단한 시스템에 효과적일 수 있습니다.

![Code vs JSON Actions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png)

## 도구 호출 에이전트는 어떻게 작동하나요?

도구 호출 에이전트는 코드 에이전트와 동일한 다단계 워크플로우를 따릅니다(자세한 내용은 [이전 섹션](./code_agents) 참조).

주요 차이점은 **액션을 구성하는 방식**에 있습니다. 실행 가능한 코드 대신, 도구 이름과 인수를 지정하는 **JSON 객체를 생성**합니다. 그런 다음 시스템은 **이 지침을 구문 분석**하여 적절한 도구를 실행합니다.

## 예시: 도구 호출 에이전트 실행하기

알프레드가 파티 준비를 시작했던 이전 예시를 다시 살펴보겠지만, 이번에는 `ToolCallingAgent`를 사용하여 차이점을 강조하겠습니다. 코드 에이전트 예시에서처럼 DuckDuckGo를 사용하여 웹을 검색할 수 있는 에이전트를 구축할 것입니다. 유일한 차이점은 에이전트 유형이며, 프레임워크가 나머지를 모두 처리합니다.

```python
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, InferenceClientModel

agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=InferenceClientModel())

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
```

에이전트의 추적을 검사하면 'Executing parsed code:' 대신 다음과 같은 내용을 볼 수 있습니다.

```text
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'web_search' with arguments: {'query': "best music recommendations for a party at Wayne's         │
│ mansion"}                                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

에이전트는 시스템이 출력을 생성하기 위해 처리하는 구조화된 도구 호출을 생성하며, `CodeAgent`처럼 코드를 직접 실행하지 않습니다.

이제 두 가지 에이전트 유형을 모두 이해했으므로 필요에 맞는 올바른 에이전트를 선택할 수 있습니다. 알프레드의 파티를 성공적으로 만들기 위해 `smolagents`를 계속 탐색해 봅시다! 🎉

## 자료

- [ToolCallingAgent documentation](https://huggingface.co/docs/smolagents/v1.8.1/en/reference/agents#smolagents.ToolCallingAgent) - ToolCallingAgent 공식 문서