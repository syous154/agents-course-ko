# 갈라 에이전트 만들기

이제 알프레드에 필요한 모든 구성 요소를 만들었으므로, 호화로운 갈라를 주최하는 데 도움이 될 완전한 에이전트로 모든 것을 통합할 차례입니다.

이 섹션에서는 게스트 정보 검색, 웹 검색, 날씨 정보 및 허브 통계 도구를 단일 강력한 에이전트로 결합합니다.

## 알프레드 조립: 완전한 에이전트

이전 섹션에서 만든 모든 도구를 다시 구현하는 대신 `tools.py` 및 `retriever.py` 파일에 저장한 각 모듈에서 가져옵니다.

<Tip>
아직 도구를 구현하지 않았다면 <a href="./tools">도구</a> 및 <a href="./invitees">리트리버</a> 섹션으로 돌아가서 구현하고 <code>tools.py</code> 및 <code>retriever.py</code> 파일에 추가하십시오.
</Tip>

이전 섹션에서 필요한 라이브러리와 도구를 가져오겠습니다.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
# 필요한 라이브러리 가져오기
import random
from smolagents import CodeAgent, InferenceClientModel

# 모듈에서 사용자 지정 도구 가져오기
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset
```

이제 이 모든 도구를 단일 에이전트로 결합해 보겠습니다.

```python
# Hugging Face 모델 초기화
model = InferenceClientModel()

# 웹 검색 도구 초기화
search_tool = DuckDuckGoSearchTool()

# 날씨 도구 초기화
weather_info_tool = WeatherInfoTool()

# 허브 통계 도구 초기화
hub_stats_tool = HubStatsTool()

# 게스트 데이터 세트를 로드하고 게스트 정보 도구 초기화
guest_info_tool = load_guest_dataset()

# 모든 도구를 사용하여 알프레드 만들기
alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True, # 추가 기본 도구 추가
    planning_interval=3 # 3단계마다 계획 활성화
)
```

</hfoption>
<hfoption id="llama-index">

```python
# 필요한 라이브러리 가져오기
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from tools import search_tool, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool
```

이제 이 모든 도구를 단일 에이전트로 결합해 보겠습니다.

```python
# Hugging Face 모델 초기화
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# 모든 도구를 사용하여 알프레드 만들기
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    llm=llm,
)
```

</hfoption>
<hfoption id="langgraph">

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from tools import DuckDuckGoSearchRun, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool
```

이제 이 모든 도구를 단일 에이전트로 결합해 보겠습니다.

```python
# 웹 검색 도구 초기화
search_tool = DuckDuckGoSearchRun()

# 도구를 포함한 채팅 인터페이스 생성
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = chat.bind_tools(tools)

# AgentState 및 Agent 그래프 생성
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

## 그래프
builder = StateGraph(AgentState)

# 노드 정의: 작업을 수행합니다.
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# 엣지 정의: 제어 흐름이 어떻게 이동하는지 결정합니다.
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # 최신 메시지에 도구가 필요한 경우 도구로 라우팅
    # 그렇지 않으면 직접 응답 제공
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()
```
</hfoption>
</hfoptions>

이제 에이전트를 사용할 준비가 되었습니다!

## 알프레드 사용: 엔드투엔드 예제

이제 알프레드가 필요한 모든 도구를 완벽하게 갖추었으므로 갈라 동안 다양한 작업을 어떻게 도울 수 있는지 살펴보겠습니다.

### 예제 1: 게스트 정보 찾기

알프레드가 게스트 정보를 어떻게 도울 수 있는지 살펴보겠습니다.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
query = "'레이디 에이다 러브레이스'에 대해 알려주세요."
response = alfred.run(query)

print("🎩 알프레드의 응답:")
print(response)
```

예상 출력:

```
🎩 알프레드의 응답:
제가 검색한 정보에 따르면, 레이디 에이다 러브레이스는 존경받는 수학자이자 친구입니다. 그녀는 찰스 배비지의 분석 엔진에 대한 연구로 최초의 컴퓨터 프로그래머로 종종 칭송받는 수학 및 컴퓨팅 분야의 선구적인 업적으로 유명합니다. 그녀의 이메일 주소는 ada.lovelace@example.com입니다.
```

</hfoption>
<hfoption id="llama-index">

```python
query = "레이디 에이다 러브레이스에 대해 알려주세요. 그녀의 배경은 무엇인가요?"
response = await alfred.run(query)

print("🎩 알프레드의 응답:")
print(response.response.blocks[0].text)
```

예상 출력:

```
🎩 알프레드의 응답:
레이디 에이다 러브레이스는 영국의 수학자이자 작가로, 찰스 배비지의 분석 엔진에 대한 연구로 가장 잘 알려져 있습니다. 그녀는 기계가 순수한 계산을 넘어선 응용 분야를 가지고 있다는 것을 처음으로 인식했습니다.
```

</hfoption>
<hfoption id="langgraph">

```python
response = alfred.invoke({"messages": "'레이디 에이다 러브레이스'에 대해 알려주세요."})

print("🎩 알프레드의 응답:")
print(response['messages'][-1].content)
```

예상 출력:

```
🎩 알프레드의 응답:
러브레이스 백작 부인 오거스타 에이다 킹으로도 알려진 에이다 러브레이스는 영국의 수학자이자 작가였습니다. 1815년 12월 10일에 태어나 1852년 11월 27일에 사망한 그녀는 제안된 기계식 범용 컴퓨터인 찰스 배비지의 분석 엔진에 대한 연구로 유명합니다. 에이다 러브레이스는 1843년에 분석 엔진을 위한 프로그램을 만들었기 때문에 최초의 컴퓨터 프로그래머 중 한 명으로 칭송받습니다. 그녀는 기계가 단순한 계산 이상의 용도로 사용될 수 있다는 것을 인식했으며, 당시에는 거의 아무도 하지 못했던 방식으로 그 잠재력을 구상했습니다. 컴퓨터 과학 분야에 대한 그녀의 공헌은 미래 발전의 토대를 마련했습니다. 10월의 어느 날, 에이다 러브레이스의 선구적인 업적에 영감을 받아 과학 기술에 대한 여성의 공헌을 기리는 에이다 러브레이스의 날로 지정되었습니다.
```

</hfoption>
</hfoptions>


### 예제 2: 불꽃놀이를 위한 날씨 확인

알프레드가 날씨를 어떻게 도울 수 있는지 살펴보겠습니다.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
query = "오늘 밤 파리의 날씨는 어떤가요? 불꽃놀이에 적합할까요?"
response = alfred.run(query)

print("🎩 알프레드의 응답:")
print(response)
```

예상 출력(무작위성으로 인해 달라질 수 있음):
```
🎩 알프레드의 응답:
파리의 날씨를 확인해 드렸습니다. 현재 맑고 기온은 25°C입니다. 이러한 조건은 오늘 밤 불꽃놀이에 완벽합니다. 맑은 하늘은 멋진 쇼를 위한 훌륭한 가시성을 제공할 것이며, 쾌적한 온도는 손님들이 불편함 없이 야외 행사를 즐길 수 있도록 보장할 것입니다.
```

</hfoption>
<hfoption id="llama-index">

```python
query = "오늘 밤 파리의 날씨는 어떤가요? 불꽃놀이에 적합할까요?"
response = await alfred.run(query)

print("🎩 알프레드의 응답:")
print(response)
```

예상 출력:

```
🎩 알프레드의 응답:
오늘 밤 파리의 날씨는 비가 오고 기온은 15°C입니다. 비가 오는 것을 감안할 때 불꽃놀이에는 적합하지 않을 수 있습니다.
```

</hfoption>
<hfoption id="langgraph">

```python
response = alfred.invoke({"messages": "오늘 밤 파리의 날씨는 어떤가요? 불꽃놀이에 적합할까요?"})

print("🎩 알프레드의 응답:")
print(response['messages'][-1].content)
```

예상 출력:

```
🎩 알프레드의 응답:
오늘 밤 파리의 날씨는 비가 오고 기온은 15°C이므로 불꽃놀이에는 적합하지 않을 수 있습니다.
```
</hfoption>
</hfoptions>

### 예제 3: AI 연구원에게 깊은 인상 남기기

알프레드가 AI 연구원에게 깊은 인상을 남기는 데 어떻게 도움이 되는지 살펴보겠습니다.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
query = "손님 중 한 분이 Qwen 출신입니다. 가장 인기 있는 모델에 대해 무엇을 알려주실 수 있나요?"
response = alfred.run(query)

print("🎩 알프레드의 응답:")
print(response)
```

예상 출력:

```
🎩 알프레드의 응답:
가장 인기 있는 Qwen 모델은 3,313,345회 다운로드된 Qwen/Qwen2.5-VL-7B-Instruct입니다.
```
</hfoption>
<hfoption id="llama-index">

```python
query = "손님 중 한 분이 Google 출신입니다. 가장 인기 있는 모델에 대해 무엇을 알려주실 수 있나요?"
response = await alfred.run(query)

print("🎩 알프레드의 응답:")
print(response)
```

예상 출력:

```
🎩 알프레드의 응답:
Hugging Face Hub에서 Google의 가장 인기 있는 모델은 28,546,752회 다운로드된 google/electra-base-discriminator입니다.
```

</hfoption>
<hfoption id="langgraph">

```python
response = alfred.invoke({"messages": "손님 중 한 분이 Qwen 출신입니다. 가장 인기 있는 모델에 대해 무엇을 알려주실 수 있나요?"})

print("🎩 알프레드의 응답:")
print(response['messages'][-1].content)
```

예상 출력:

```
🎩 알프레드의 응답:
Qwen에서 가장 많이 다운로드된 모델은 3,313,345회 다운로드된 Qwen/Qwen2.5-VL-7B-Instruct입니다.
```
</hfoption>
</hfoptions>

### 예제 4: 여러 도구 결합

알프레드가 니콜라 테슬라 박사와의 대화를 준비하는 데 어떻게 도움이 되는지 살펴보겠습니다.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
query = "최근 무선 에너지 발전에 대해 니콜라 테슬라 박사와 이야기해야 합니다. 이 대화를 준비하는 데 도움을 주실 수 있나요?"
response = alfred.run(query)

print("🎩 알프레드의 응답:")
print(response)
```

예상 출력:

```
🎩 알프레드의 응답:
니콜라 테슬라 박사와의 대화를 준비하는 데 도움이 될 정보를 수집했습니다.

게스트 정보:
이름: 니콜라 테슬라 박사
관계: 대학 시절의 오랜 친구
설명: 니콜라 테슬라 박사는 대학 시절의 오랜 친구입니다. 그는 최근에 새로운 무선 에너지 전송 시스템에 대한 특허를 받았으며 이에 대해 기꺼이 논의할 것입니다. 그가 비둘기에 열정적이라는 것을 기억하십시오. 그래서 그것이 좋은 잡담거리가 될 수 있습니다.
이메일: nikola.tesla@gmail.com

무선 에너지의 최근 발전:
웹 검색을 기반으로 한 무선 에너지 전송의 최근 개발 사항은 다음과 같습니다.
1. 연구원들은 집중된 전자기파를 사용하여 장거리 무선 전력 전송에 진전을 이루었습니다.
2. 여러 회사에서 가전제품을 위한 공진 유도 결합 기술을 개발하고 있습니다.
3. 물리적 연결 없이 전기 자동차 충전에 새로운 응용 분야가 있습니다.

대화 시작점:
1. "무선 에너지 전송에 대한 새로운 특허에 대해 듣고 싶습니다. 대학 시절의 원래 개념과 어떻게 비교됩니까?"
2. "가전제품을 위한 공진 유도 결합의 최근 개발을 보셨습니까? 그들의 접근 방식에 대해 어떻게 생각하십니까?"
3. "비둘기는 잘 지내나요? 그들에 대한 당신의 매력을 기억합니다."

이것은 테슬라 박사와 그의 관심사와 그의 분야의 최근 발전에 대한 지식을 보여주면서 논의할 충분한 내용을 제공해야 합니다.
```

</hfoption>
<hfoption id="llama-index">

```python
query = "최근 무선 에너지 발전에 대해 니콜라 테슬라 박사와 이야기해야 합니다. 이 대화를 준비하는 데 도움을 주실 수 있나요?"
response = await alfred.run(query)

print("🎩 알프레드의 응답:")
print(response)
```

예상 출력:

```
🎩 알프레드의 응답:
니콜라 테슬라 박사와의 대화에 유용할 수 있는 무선 에너지의 최근 발전 사항은 다음과 같습니다.

1. **무선 전력 전송의 발전과 과제**: 이 기사에서는 기존 유선 방식에서 태양광 우주 발전소와 같은 현대적인 응용 분야에 이르기까지 무선 전력 전송(WPT)의 진화를 논의합니다. 마이크로파 기술에 대한 초기 초점과 전기 장치의 증가로 인한 현재 WPT 수요를 강조합니다.

2. **신체 인터페이스 전자 장치를 위한 무선 에너지 전송 기술의 최근 발전**: 이 기사에서는 배터리나 리드선 없이 신체 인터페이스 전자 장치에 전력을 공급하는 솔루션으로 무선 에너지 전송(WET)을 탐구합니다. 이 맥락에서 WET의 장점과 잠재적인 응용 분야를 논의합니다.

3. **무선 전력 전송 및 에너지 하베스팅: 현황 및 미래 동향**: 이 기사에서는 에너지 하베스팅 및 무선 전력 전송을 포함한 최근 무선 전원 공급 방법의 발전에 대한 개요를 제공합니다. 몇 가지 유망한 응용 분야를 제시하고 해당 분야의 미래 동향을 논의합니다.

4. **무선 전력 전송: 응용, 과제, 장벽 및
```

</hfoption>
<hfoption id="langgraph">

```python
response = alfred.invoke({"messages":"최근 무선 에너지 발전에 대해 '니콜라 테슬라 박사'와 이야기해야 합니다. 이 대화를 준비하는 데 도움을 주실 수 있나요?"})

print("🎩 알프레드의 응답:")
print(response['messages'][-1].content)
```

예상 출력:

```
제공된 정보를 바탕으로, 무선 에너지의 최근 발전에 대해 '니콜라 테슬라 박사'와의 대화를 준비하기 위한 핵심 사항은 다음과 같습니다.
1. **무선 전력 전송(WPT):** WPT가 유도 및 공진 결합과 같은 메커니즘을 활용하여 코드 없이 에너지 전송을 혁신하는 방법에 대해 논의합니다.
2. **무선 충전의 발전:** 효율성 향상, 더 빠른 충전 속도, Qi/Qi2 인증 무선 충전 솔루션의 부상에 대해 강조합니다.
3. **5G-Advanced 혁신 및 NearLink 무선 프로토콜:** 무선 네트워크의 속도, 보안 및 효율성을 향상시키는 개발 사항으로 언급하며, 이는 고급 무선 에너지 기술을 지원할 수 있습니다.
4. **엣지에서의 AI 및 ML:** AI 및 머신 러닝이 무선 네트워크에 의존하여 엣지에 인텔리전스를 제공하고 스마트 홈 및 빌딩의 자동화 및 인텔리전스를 향상시키는 방법에 대해 이야기합니다.
5. **Matter, Thread 및 보안 발전:** IoT 장치 및 시스템의 연결성, 효율성 및 보안을 주도하는 핵심 혁신으로 논의합니다.
6. **무선 충전 기술의 돌파구:** 무선 충전의 발전을 입증하기 위해 인천대학교의 연구와 같은 최근의 돌파구나 연구를 포함합니다.
```
</hfoption>
</hfoptions>

## 고급 기능: 대화 메모리

갈라 동안 알프레드를 더욱 유용하게 만들기 위해 이전 상호 작용을 기억하도록 대화 메모리를 활성화할 수 있습니다.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
# 대화 메모리가 있는 알프레드 만들기
alfred_with_memory = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True,
    planning_interval=3
)

# 첫 번째 상호 작용
response1 = alfred_with_memory.run("레이디 에이다 러브레이스에 대해 알려주세요.")
print("🎩 알프레드의 첫 번째 응답:")
print(response1)

# 두 번째 상호 작용 (첫 번째 참조)
response2 = alfred_with_memory.run("그녀는 현재 어떤 프로젝트를 진행하고 있나요?", reset=False)
print("🎩 알프레드의 두 번째 응답:")
print(response2)
```

</hfoption>
<hfoption id="llama-index">

```python
from llama_index.core.workflow import Context

alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    llm=llm
)

# 상태 기억
ctx = Context(alfred)

# 첫 번째 상호 작용
response1 = await alfred.run("레이디 에이다 러브레이스에 대해 알려주세요.", ctx=ctx)
print("🎩 알프레드의 첫 번째 응답:")
print(response1)

# 두 번째 상호 작용 (첫 번째 참조)
response2 = await alfred.run("그녀는 현재 어떤 프로젝트를 진행하고 있나요?", ctx=ctx)
print("🎩 알프레드의 두 번째 응답:")
print(response2)
```

</hfoption>
<hfoption id="langgraph">

```python
# 첫 번째 상호 작용
response = alfred.invoke({"messages": [HumanMessage(content="레이디 에이다 러브레이스에 대해 알려주세요. 그녀의 배경과 저와의 관계는 무엇인가요?")]})


print("🎩 알프레드의 응답:")
print(response['messages'][-1].content)
print()

# 두 번째 상호 작용 (첫 번째 참조)
response = alfred.invoke({"messages": response["messages"] + [HumanMessage(content="그녀는 현재 어떤 프로젝트를 진행하고 있나요?")]})

print("🎩 알프레드의 응답:")
print(response['messages'][-1].content)
```

</hfoption>
</hfoptions>

이 세 가지 에이전트 접근 방식 중 어느 것도 메모리를 에이전트와 직접 결합하지 않는다는 점에 유의하십시오. 이 디자인 선택에 특별한 이유가 있습니까 🧐?
* smolagents: 메모리는 다른 실행 실행 간에 보존되지 않으므로 `reset=False`를 사용하여 명시적으로 명시해야 합니다.
* LlamaIndex: 실행 내에서 메모리 관리를 위해 컨텍스트 개체를 명시적으로 추가해야 합니다.
* LangGraph: 이전 메시지를 검색하거나 전용 [MemorySaver](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-3-adding-memory-to-the-chatbot) 구성 요소를 활용하는 옵션을 제공합니다.

## 결론

축하합니다! 세기의 가장 호화로운 갈라를 주최하는 데 도움이 되는 여러 도구를 갖춘 정교한 에이전트인 알프레드를 성공적으로 구축했습니다. 이제 알프레드는 다음을 수행할 수 있습니다.

1. 손님에 대한 자세한 정보 검색
2. 야외 활동 계획을 위한 기상 조건 확인
3. 영향력 있는 AI 빌더 및 해당 모델에 대한 통찰력 제공
4. 최신 정보를 위한 웹 검색
5. 메모리로 대화 컨텍스트 유지

이러한 기능을 통해 알프레드는 개인화된 관심과 최신 정보로 손님에게 깊은 인상을 남겨 갈라가 큰 성공을 거두도록 보장할 준비가 되었습니다.