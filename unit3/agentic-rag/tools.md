# 에이전트를 위한 도구 구축 및 통합

이 섹션에서는 알프레드에게 웹 액세스 권한을 부여하여 최신 뉴스와 글로벌 업데이트를 찾을 수 있도록 합니다.
또한 날씨 데이터와 Hugging Face 허브 모델 다운로드 통계에 액세스하여 새로운 주제에 대한 관련 대화를 나눌 수 있습니다.

## 에이전트에게 웹 액세스 권한 부여

알프레드가 세상에 대한 깊은 지식을 가진 진정한 르네상스 호스트로서의 입지를 확립하기를 원한다는 것을 기억하십시오.

이를 위해 알프레드가 세상에 대한 최신 뉴스와 정보에 액세스할 수 있도록 해야 합니다.

알프레드를 위한 웹 검색 도구를 만드는 것부터 시작하겠습니다!

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
from smolagents import DuckDuckGoSearchTool

# DuckDuckGo 검색 도구 초기화
search_tool = DuckDuckGoSearchTool()

# 사용 예시
results = search_tool("현재 프랑스 대통령은 누구입니까?")
print(results)
```

예상 출력:

```
현재 프랑스 대통령은 에마뉘엘 마크롱입니다.
```


</hfoption>
<hfoption id="llama-index">

```python
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.tools import FunctionTool

# DuckDuckGo 검색 도구 초기화
tool_spec = DuckDuckGoSearchToolSpec()

search_tool = FunctionTool.from_defaults(tool_spec.duckduckgo_full_search)
# 사용 예시
response = search_tool("현재 프랑스 대통령은 누구입니까?")
print(response.raw_output[-1]['body'])
```

예상 출력:

```
프랑스 공화국 대통령은 프랑스의 국가 원수입니다. 현 대통령은 2017년 5월 14일부터 에마뉘엘 마크롱이며, 2017년 5월 7일 대통령 선거 2차 투표에서 마린 르펜을 꺾었습니다. 프랑스 대통령 목록(제5공화국) N° 초상화 이름 ...
```

</hfoption>
<hfoption id="langgraph">

```python
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke("현재 프랑스 대통령은 누구입니까?")
print(results)
```

예상 출력:

```
에마뉘엘 마크롱(1977년 12월 21일 프랑스 아미앵 출생)은 프랑스의 은행가이자 정치인으로 2017년 프랑스 대통령으로 선출되었습니다...
```

</hfoption>
</hfoptions>

## 불꽃놀이 일정을 위한 날씨 정보 사용자 지정 도구 만들기

완벽한 갈라에는 맑은 하늘 위로 불꽃놀이가 펼쳐져야 하므로, 악천후로 인해 불꽃놀이가 취소되지 않도록 해야 합니다.

외부 날씨 API를 호출하여 특정 위치의 날씨 정보를 가져오는 데 사용할 수 있는 사용자 지정 도구를 만들어 보겠습니다.

<Tip>
간단하게 하기 위해 이 예제에서는 더미 날씨 API를 사용하고 있습니다. 실제 날씨 API를 사용하려면 <a href="../../unit1/tutorial">1단원</a>에서처럼 OpenWeatherMap API를 사용하는 날씨 도구를 구현할 수 있습니다.
</Tip>

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
from smolagents import Tool
import random

class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "특정 위치에 대한 더미 날씨 정보를 가져옵니다."
    inputs = {
        "location": {
            "type": "string",
            "description": "날씨 정보를 가져올 위치입니다."
        }
    }
    output_type = "string"

    def forward(self, location: str):
        # 더미 날씨 데이터
        weather_conditions = [
            {"condition": "비", "temp_c": 15},
            {"condition": "맑음", "temp_c": 25},
            {"condition": "바람", "temp_c": 20}
        ]
        # 날씨 상태를 무작위로 선택
        data = random.choice(weather_conditions)
        return f"{location}의 날씨: {data['condition']}, {data['temp_c']}°C"

# 도구 초기화
weather_info_tool = WeatherInfoTool()
```

</hfoption>
<hfoption id="llama-index">

```python
import random
from llama_index.core.tools import FunctionTool

def get_weather_info(location: str) -> str:
    """특정 위치에 대한 더미 날씨 정보를 가져옵니다."""
    # 더미 날씨 데이터
    weather_conditions = [
        {"condition": "비", "temp_c": 15},
        {"condition": "맑음", "temp_c": 25},
        {"condition": "바람", "temp_c": 20}
    ]
    # 날씨 상태를 무작위로 선택
    data = random.choice(weather_conditions)
    return f"{location}의 날씨: {data['condition']}, {data['temp_c']}°C"

# 도구 초기화
weather_info_tool = FunctionTool.from_defaults(get_weather_info)
```

</hfoption>
<hfoption id="langgraph">

```python
from langchain.tools import Tool
import random

def get_weather_info(location: str) -> str:
    """특정 위치에 대한 더미 날씨 정보를 가져옵니다."""
    # 더미 날씨 데이터
    weather_conditions = [
        {"condition": "비", "temp_c": 15},
        {"condition": "맑음", "temp_c": 25},
        {"condition": "바람", "temp_c": 20}
    ]
    # 날씨 상태를 무작위로 선택
    data = random.choice(weather_conditions)
    return f"{location}의 날씨: {data['condition']}, {data['temp_c']}°C"

# 도구 초기화
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="특정 위치에 대한 더미 날씨 정보를 가져옵니다."
)
```

</hfoption>
</hfoptions>

## 영향력 있는 AI 빌더를 위한 허브 통계 도구 만들기

갈라에는 AI 빌더의 거물들이 참석합니다. 알프레드는 가장 인기 있는 모델, 데이터 세트 및 공간에 대해 논의하여 그들에게 깊은 인상을 남기고 싶어합니다. 사용자 이름을 기반으로 Hugging Face 허브에서 모델 통계를 가져오는 도구를 만들 것입니다.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
from smolagents import Tool
from huggingface_hub import list_models

class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Hugging Face 허브에서 특정 작성자의 가장 많이 다운로드된 모델을 가져옵니다."
    inputs = {
        "author": {
            "type": "string",
            "description": "모델을 찾을 모델 작성자/조직의 사용자 이름입니다."
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            # 지정된 작성자의 모델을 다운로드 순으로 정렬하여 나열
            models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))
            
            if models:
                model = models[0]
                return f"{author}의 가장 많이 다운로드된 모델은 {model.id}이며 다운로드 수는 {model.downloads:,}입니다."
            else:
                return f"{author}에 대한 모델을 찾을 수 없습니다."
        except Exception as e:
            return f"{author}에 대한 모델을 가져오는 중 오류 발생: {str(e)}"

# 도구 초기화
hub_stats_tool = HubStatsTool()

# 사용 예시
print(hub_stats_tool("facebook")) # 예시: Facebook에서 가장 많이 다운로드된 모델 가져오기
```

예상 출력:

```
facebook에서 가장 많이 다운로드된 모델은 facebook/esmfold_v1이며 다운로드 수는 12,544,550입니다.
```

</hfoption>
<hfoption id="llama-index">

```python
import random
from llama_index.core.tools import FunctionTool
from huggingface_hub import list_models

def get_hub_stats(author: str) -> str:
    """Hugging Face 허브에서 특정 작성자의 가장 많이 다운로드된 모델을 가져옵니다."""
    try:
        # 지정된 작성자의 모델을 다운로드 순으로 정렬하여 나열
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"{author}의 가장 많이 다운로드된 모델은 {model.id}이며 다운로드 수는 {model.downloads:,}입니다."
        else:
            return f"{author}에 대한 모델을 찾을 수 없습니다."
    except Exception as e:
        return f"{author}에 대한 모델을 가져오는 중 오류 발생: {str(e)}"

# 도구 초기화
hub_stats_tool = FunctionTool.from_defaults(get_hub_stats)

# 사용 예시
print(hub_stats_tool("facebook")) # 예시: Facebook에서 가장 많이 다운로드된 모델 가져오기
```

예상 출력:

```
facebook에서 가장 많이 다운로드된 모델은 facebook/esmfold_v1이며 다운로드 수는 12,544,550입니다.
```

</hfoption>
<hfoption id="langgraph">

```python
from langchain.tools import Tool
from huggingface_hub import list_models

def get_hub_stats(author: str) -> str:
    """Hugging Face 허브에서 특정 작성자의 가장 많이 다운로드된 모델을 가져옵니다."""
    try:
        # 지정된 작성자의 모델을 다운로드 순으로 정렬하여 나열
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"{author}의 가장 많이 다운로드된 모델은 {model.id}이며 다운로드 수는 {model.downloads:,}입니다."
        else:
            return f"{author}에 대한 모델을 찾을 수 없습니다."
    except Exception as e:
        return f"{author}에 대한 모델을 가져오는 중 오류 발생: {str(e)}"

# 도구 초기화
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Hugging Face 허브에서 특정 작성자의 가장 많이 다운로드된 모델을 가져옵니다."
)

# 사용 예시
print(hub_stats_tool.invoke("facebook")) # 예시: Facebook에서 가장 많이 다운로드된 모델 가져오기
```

예상 출력:

```
facebook에서 가장 많이 다운로드된 모델은 facebook/esmfold_v1이며 다운로드 수는 13,109,861입니다.
```

</hfoption>
</hfoptions>

허브 통계 도구를 사용하면 알프레드는 이제 가장 인기 있는 모델에 대해 논의하여 영향력 있는 AI 빌더에게 깊은 인상을 남길 수 있습니다.

## 알프레드와 도구 통합

이제 모든 도구가 있으므로 알프레드의 에이전트에 통합해 보겠습니다.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
from smolagents import CodeAgent, InferenceClientModel

# Hugging Face 모델 초기화
model = InferenceClientModel()

# 모든 도구를 사용하여 알프레드 만들기
alfred = CodeAgent(
    tools=[search_tool, weather_info_tool, hub_stats_tool], 
    model=model
)

# 갈라 동안 알프레드가 받을 수 있는 예시 쿼리
response = alfred.run("Facebook은 무엇이며 가장 인기 있는 모델은 무엇입니까?")

print("🎩 알프레드의 응답:")
print(response)
```

예상 출력:

```
🎩 알프레드의 응답:
Facebook은 사용자가 연결하고 정보를 공유하며 다른 사람들과 상호 작용할 수 있는 소셜 네트워킹 웹사이트입니다. Hugging Face 허브에서 Facebook이 가장 많이 다운로드한 모델은 ESMFold_v1입니다.
```

</hfoption>
<hfoption id="llama-index">

```python
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# Hugging Face 모델 초기화
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
# 모든 도구를 사용하여 알프레드 만들기
alfred = AgentWorkflow.from_tools_or_functions(
    [search_tool, weather_info_tool, hub_stats_tool],
    llm=llm
)

# 갈라 동안 알프레드가 받을 수 있는 예시 쿼리
response = await alfred.run("Facebook은 무엇이며 가장 인기 있는 모델은 무엇입니까?")

print("🎩 알프레드의 응답:")
print(response)
```

예상 출력:

```
🎩 알프레드의 응답:
Facebook은 캘리포니아 멘로 파크에 본사를 둔 소셜 네트워킹 서비스 및 기술 회사입니다. 마크 저커버그가 설립했으며 사람들이 프로필을 만들고 친구 및 가족과 연결하고 사진과 비디오를 공유하고 공통 관심사를 기반으로 그룹에 가입할 수 있습니다. Hugging Face 허브에서 Facebook의 가장 인기 있는 모델은 `facebook/esmfold_v1`이며 다운로드 수는 13,109,861입니다.
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

# 도구를 포함한 채팅 인터페이스 생성
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [search_tool, weather_info_tool, hub_stats_tool]
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

messages = [HumanMessage(content="Facebook은 누구이며 가장 인기 있는 모델은 무엇입니까?")]
response = alfred.invoke({"messages": messages})

print("🎩 알프레드의 응답:")
print(response['messages'][-1].content)
```

예상 출력:

```
🎩 알프레드의 응답:
Facebook은 소셜 네트워킹 사이트인 Facebook과 Instagram 및 WhatsApp과 같은 기타 서비스로 유명한 소셜 미디어 회사입니다. Hugging Face 허브에서 Facebook이 가장 많이 다운로드한 모델은 facebook/esmfold_v1이며 다운로드 수는 13,202,321입니다.
```
</hfoption>
</hfoptions>

## 결론

이러한 도구를 통합함으로써 알프레드는 이제 웹 검색에서 날씨 업데이트 및 모델 통계에 이르기까지 다양한 작업을 처리할 수 있는 능력을 갖추게 되었습니다. 이를 통해 그는 갈라에서 가장 정보에 밝고 매력적인 호스트로 남을 수 있습니다.

<Tip>
특정 주제에 대한 최신 뉴스를 가져오는 데 사용할 수 있는 도구를 구현해 보십시오.

완료되면 <code>tools.py</code> 파일에 사용자 지정 도구를 구현하십시오.
</Tip>


이제 도구가 있으므로 다음 섹션으로 이동하여 손님 이야기를 위한 RAG 도구를 만들 것입니다.