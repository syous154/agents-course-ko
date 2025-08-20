# 문서 분석 그래프

집사 알프레드가 인사드립니다. 웨인 씨의 충실한 집사로서, 저는 웨인 씨의 다양한 문서 관련 필요를 어떻게 돕는지 기록하는 특권을 누렸습니다. 그분이 밤 활동에 몰두하시는 동안, 저는 모든 서류, 훈련 일정, 영양 계획이 적절하게 분석되고 정리되도록 합니다.

떠나기 전에, 그분은 한 주간의 훈련 프로그램이 담긴 쪽지를 남기셨습니다. 저는 그 후 내일 식사를 위한 **메뉴**를 고안하는 책임을 맡았습니다.

미래의 이러한 이벤트를 위해, 웨인 씨의 필요를 충족시키기 위해 LangGraph를 사용하여 문서 분석 시스템을 만들어 봅시다. 이 시스템은 다음을 수행할 수 있습니다:

1. 이미지 문서 처리
2. 비전 모델(Vision Language Model)을 사용하여 텍스트 추출
3. 필요할 때 계산 수행 (일반 도구 시연용)
4. 콘텐츠 분석 및 간결한 요약 제공
5. 문서와 관련된 특정 지침 실행

## 집사의 워크플로우

우리가 구축할 워크플로우는 다음 구조화된 스키마를 따릅니다:

![집사의 문서 분석 워크플로우](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/alfred_flow.png)

<Tip>
<a href="https://huggingface.co/datasets/agents-course/notebooks/blob/main/unit2/langgraph/agent.ipynb" target="_blank">이 노트북</a>에서 코드를 따라할 수 있으며, Google Colab을 사용하여 실행할 수 있습니다.
</Tip>

## 환경 설정

```python
%pip install langgraph langchain_openai langchain_core
```
및 임포트:
```python
import base64
from typing import List, TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
```

## 에이전트 상태 정의

이 상태는 이전에 보았던 것보다 조금 더 복잡합니다.
`AnyMessage`는 메시지를 정의하는 Langchain의 클래스이며, `add_messages`는 최신 상태로 덮어쓰는 대신 최신 메시지를 추가하는 연산자입니다.

이는 LangGraph의 새로운 개념으로, 상태에 연산자를 추가하여 상호 작용 방식을 정의할 수 있습니다.

```python
class AgentState(TypedDict):
    # 제공된 문서
    input_file: Optional[str]  # 파일 경로 (PDF/PNG) 포함
    messages: Annotated[list[AnyMessage], add_messages]
```

## 도구 준비

```python
vision_llm = ChatOpenAI(model="gpt-4o")

def extract_text(img_path: str) -> str:
    """
    다중 모달 모델을 사용하여 이미지 파일에서 텍스트를 추출합니다.
    
    웨인 주인님은 종종 훈련 계획이나 식사 계획이 담긴 쪽지를 남기십니다.
    이를 통해 제가 내용을 적절하게 분석할 수 있습니다.
    """
    all_text = ""
    try:
        # 이미지를 읽고 base64로 인코딩
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # base64 이미지 데이터를 포함한 프롬프트 준비
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "이 이미지에서 모든 텍스트를 추출하세요. "
                            "추출된 텍스트만 반환하고 설명은 하지 마세요."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # 비전 가능 모델 호출
        response = vision_llm.invoke(message)

        # 추출된 텍스트 추가
        all_text += response.content + "\n\n"

        return all_text.strip()
    except Exception as e:
        # 집사는 오류를 우아하게 처리해야 합니다.
        error_msg = f"텍스트 추출 오류: {str(e)}"
        print(error_msg)
        return ""

def divide(a: int, b: int) -> float:
    """a를 b로 나눕니다 - 웨인 주인님의 가끔 있는 계산을 위해."""
    return a / b

# 집사에게 도구 장착
tools = [
    divide,
    extract_text
]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
```

## 노드

```python
def assistant(state: AgentState):
    # 시스템 메시지
    textual_description_of_tool="""
extract_text(img_path: str) -> str:
    다중 모달 모델을 사용하여 이미지 파일에서 텍스트를 추출합니다.

    인수:
        img_path: 로컬 이미지 파일 경로 (문자열).

    반환:
        각 이미지에서 추출된 텍스트를 연결한 단일 문자열.
divide(a: int, b: int) -> float:
    a를 b로 나눕니다.
"""
    image=state["input_file"]
    sys_msg = SystemMessage(content=f"당신은 웨인 씨와 배트맨을 모시는 유능한 집사 알프레드입니다. 제공된 도구를 사용하여 문서를 분석하고 계산을 실행할 수 있습니다:\n{textual_description_of_tool} \n 일부 선택적 이미지에 접근할 수 있습니다. 현재 로드된 이미지는: {image}")

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"]
    }
```

## ReAct 패턴: 제가 웨인 씨를 돕는 방법

이 에이전트의 접근 방식을 설명해 드리겠습니다. 이 에이전트는 ReAct 패턴(Reason-Act-Observe)으로 알려진 것을 따릅니다.

1. 문서와 요청에 대해 **추론**합니다.
2. 적절한 도구를 사용하여 **행동**합니다.
3. 결과를 **관찰**합니다.
4. 필요를 완전히 해결할 때까지 필요에 따라 **반복**합니다.

이것은 LangGraph를 사용한 에이전트의 간단한 구현입니다.

```python
# 그래프
builder = StateGraph(AgentState)

# 노드 정의: 이들이 작업을 수행합니다.
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# 엣지 정의: 이들이 제어 흐름이 어떻게 이동하는지 결정합니다.
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # 최신 메시지가 도구를 필요로 하면 도구로 라우팅합니다.
    # 그렇지 않으면 직접 응답을 제공합니다.
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# 집사의 사고 과정 표시
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
```

도구 목록이 있는 `tools` 노드를 정의합니다. `assistant` 노드는 바인딩된 도구가 있는 모델입니다.
`assistant` 및 `tools` 노드로 그래프를 생성합니다.

`tools_condition` 엣지를 추가하여, `assistant`가 도구를 호출하는지 여부에 따라 `End` 또는 `tools`로 라우팅합니다.

이제 새로운 단계를 하나 더 추가합니다:

`tools` 노드를 `assistant`로 다시 연결하여 루프를 형성합니다.

- `assistant` 노드가 실행된 후, `tools_condition`은 모델의 출력이 도구 호출인지 확인합니다.
- 도구 호출인 경우, 흐름은 `tools` 노드로 전달됩니다.
- `tools` 노드는 `assistant`로 다시 연결됩니다.
- 이 루프는 모델이 도구를 호출하기로 결정하는 한 계속됩니다.
- 모델 응답이 도구 호출이 아니면 흐름은 END로 전달되어 프로세스가 종료됩니다.

![ReAct 패턴](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/Agent.png)

## 행동하는 집사

### 예시 1: 간단한 계산

다음은 LangGraph에서 도구를 사용하는 에이전트의 간단한 사용 사례를 보여주는 예시입니다.

```python
messages = [HumanMessage(content="6790을 5로 나누세요")]
messages = react_graph.invoke({"messages": messages, "input_file": None})

# 메시지 표시
for m in messages['messages']:
    m.pretty_print()
```

대화는 다음과 같이 진행됩니다:

```
Human: 6790을 5로 나누세요

AI 도구 호출: divide(a=6790, b=5)

도구 응답: 1358.0

알프레드: 6790을 5로 나눈 결과는 1358.0입니다.
```

### 예시 2: 웨인 주인님의 훈련 문서 분석

웨인 주인님이 훈련 및 식사 쪽지를 남기실 때:

```python
messages = [HumanMessage(content="웨인 씨가 제공한 이미지의 쪽지에 따르면, 저녁 식사 메뉴를 위해 제가 사야 할 품목 목록은 무엇입니까?")]
messages = react_graph.invoke({"messages": messages, "input_file": "Batman_training_and_meals.png"})
```

상호 작용은 다음과 같이 진행됩니다:

```
Human: 웨인 씨가 제공한 이미지의 쪽지에 따르면, 저녁 식사 메뉴를 위해 제가 사야 할 품목 목록은 무엇입니까?

AI 도구 호출: extract_text(img_path="Batman_training_and_meals.png")

도구 응답: [훈련 일정 및 메뉴 세부 정보가 포함된 추출된 텍스트]

알프레드: 저녁 식사 메뉴를 위해 다음 품목을 구매하셔야 합니다:

1. 목초 사육 채끝 스테이크
2. 유기농 시금치
3. 피킬로 고추
4. 감자 (오븐에 구운 황금 허브 감자용)
5. 어유 (2그램)

최고 품질의 식사를 위해 스테이크는 목초 사육된 것이고 시금치와 고추는 유기농인지 확인하십시오.
```

## 핵심 요점

자신만의 문서 분석 집사를 만들고 싶다면, 다음 핵심 고려 사항을 참고하십시오:

1. 특정 문서 관련 작업을 위한 **명확한 도구 정의**
2. 도구 호출 간에 컨텍스트를 유지하기 위한 **견고한 상태 추적기 생성**
3. 도구 실패에 대한 **오류 처리 고려**
4. 이전 상호 작용에 대한 **맥락적 인식 유지** (`add_messages` 연산자에 의해 보장됨)

이러한 원칙을 통해 당신도 웨인 저택에 걸맞은 훌륭한 문서 분석 서비스를 제공할 수 있습니다.

*이 설명이 만족스러웠기를 바랍니다. 이제 실례하겠습니다. 오늘 밤 활동 전에 웨인 주인님의 망토를 다려야 합니다.*
