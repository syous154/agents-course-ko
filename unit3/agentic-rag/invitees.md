# 손님 이야기를 위한 RAG 도구 만들기


신뢰할 수 있는 에이전트인 알프레드는 세기의 가장 호화로운 갈라를 준비하고 있습니다. 행사가 원활하게 진행되도록 알프레드는 각 손님에 대한 최신 정보에 신속하게 액세스해야 합니다. 사용자 지정 데이터 세트로 구동되는 사용자 지정 검색 증강 생성(RAG) 도구를 만들어 알프레드를 도와줍시다.

## 갈라에 RAG를 사용하는 이유

알프레드가 손님들 사이를 오가며 각 사람에 대한 특정 세부 정보를 즉시 기억해야 한다고 상상해 보십시오. 기존 LLM은 다음과 같은 이유로 이 작업에 어려움을 겪을 수 있습니다.

1. 손님 목록은 이벤트에 따라 다르며 모델의 학습 데이터에 없습니다.
2. 손님 정보는 자주 변경되거나 업데이트될 수 있습니다.
3. 알프레드는 이메일 주소와 같은 정확한 세부 정보를 검색해야 합니다.

이것이 바로 검색 증강 생성(RAG)이 빛을 발하는 부분입니다! 검색 시스템을 LLM과 결합함으로써 알프레드는 필요할 때 손님에 대한 정확하고 최신 정보에 액세스할 수 있습니다.

<Tip>

이 사용 사례에 대해 과정에서 다루는 프레임워크 중 하나를 선택할 수 있습니다. 코드 탭에서 선호하는 옵션을 선택하십시오.

</Tip>

## 애플리케이션 설정

이 단원에서는 구조화된 Python 프로젝트로 HF Space 내에서 에이전트를 개발합니다. 이 접근 방식은 다양한 기능을 별도의 파일로 구성하여 깨끗하고 모듈화된 코드를 유지하는 데 도움이 됩니다. 또한 이는 애플리케이션을 공개적으로 배포하는 보다 현실적인 사용 사례를 만듭니다.

### 프로젝트 구조

- **`tools.py`** – 에이전트를 위한 보조 도구를 제공합니다.
- **`retriever.py`** – 지식 액세스를 지원하기 위해 검색 기능을 구현합니다.
- **`app.py`** – 모든 구성 요소를 이 단원의 마지막 부분에서 완성할 완전한 기능의 에이전트로 통합합니다.

실습 참조를 위해 이 단원에서 개발된 에이전트 RAG가 라이브인 [이 HF Space](https://huggingface.co/spaces/agents-course/Unit_3_Agentic_RAG)를 확인하십시오. 자유롭게 복제하고 실험해 보십시오!

아래에서 에이전트를 직접 테스트할 수 있습니다.

<iframe
	src="https://agents-course-unit-3-agentic-rag.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

## 데이터 세트 개요

데이터 세트 [`agents-course/unit3-invitees`](https://huggingface.co/datasets/agents-course/unit3-invitees/)에는 각 손님에 대한 다음 필드가 포함되어 있습니다.

- **이름**: 손님의 전체 이름
- **관계**: 손님이 호스트와 어떻게 관련되어 있는지
- **설명**: 손님에 대한 간략한 전기 또는 흥미로운 사실
- **이메일 주소**: 초대장 또는 후속 조치를 보내기 위한 연락처 정보

아래는 데이터 세트 미리보기입니다.
<iframe
  src="https://huggingface.co/datasets/agents-course/unit3-invitees/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

<Tip>
실제 시나리오에서 이 데이터 세트는 식이 선호도, 선물 관심사, 피해야 할 대화 주제 및 호스트에게 유용한 기타 세부 정보를 포함하도록 확장될 수 있습니다.
</Tip>

## 방명록 도구 만들기

알프레드가 갈라 동안 손님 정보를 신속하게 검색하는 데 사용할 수 있는 사용자 지정 도구를 만들 것입니다. 이를 세 가지 관리 가능한 단계로 나누겠습니다.

1. 데이터 세트 로드 및 준비
2. 리트리버 도구 만들기
3. 알프레드와 도구 통합

데이터 세트 로드 및 준비부터 시작하겠습니다!

### 1단계: 데이터 세트 로드 및 준비

먼저 원시 손님 데이터를 검색에 최적화된 형식으로 변환해야 합니다.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

Hugging Face `datasets` 라이브러리를 사용하여 데이터 세트를 로드하고 `langchain.docstore.document` 모듈의 `Document` 개체 목록으로 변환합니다.

```python
import datasets
from langchain_core.documents import Document

# 데이터 세트 로드
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# 데이터 세트 항목을 Document 개체로 변환
docs = [
    Document(
        page_content="
".join([
            f"이름: {guest['name']}",
            f"관계: {guest['relation']}",
            f"설명: {guest['description']}",
            f"이메일: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]

```

</hfoption>
<hfoption id="llama-index">

Hugging Face `datasets` 라이브러리를 사용하여 데이터 세트를 로드하고 `llama_index.core.schema` 모듈의 `Document` 개체 목록으로 변환합니다.

```python
import datasets
from llama_index.core.schema import Document

# 데이터 세트 로드
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# 데이터 세트 항목을 Document 개체로 변환
docs = [
    Document(
        text="
".join([
            f"이름: {guest_dataset['name'][i]}",
            f"관계: {guest_dataset['relation'][i]}",
            f"설명: {guest_dataset['description'][i]}",
            f"이메일: {guest_dataset['email'][i]}"
        ]),
        metadata={"name": guest_dataset['name'][i]}
    )
    for i in range(len(guest_dataset))
]
```

</hfoption>
<hfoption id="langgraph">

Hugging Face `datasets` 라이브러리를 사용하여 데이터 세트를 로드하고 `langchain.docstore.document` 모듈의 `Document` 개체 목록으로 변환합니다.

```python
import datasets
from langchain_core.documents import Document

# 데이터 세트 로드
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# 데이터 세트 항목을 Document 개체로 변환
docs = [
    Document(
        page_content="
".join([
            f"이름: {guest['name']}",
            f"관계: {guest['relation']}",
            f"설명: {guest['description']}",
            f"이메일: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]
```

</hfoption>
</hfoptions>

위 코드에서는 다음을 수행합니다.
- 데이터 세트 로드
- 각 손님 항목을 서식이 지정된 콘텐츠가 있는 `Document` 개체로 변환
- `Document` 개체를 목록에 저장

즉, 검색을 구성하기 시작할 수 있도록 모든 데이터를 멋지게 사용할 수 있습니다.

### 2단계: 리트리버 도구 만들기

이제 알프레드가 손님 정보를 검색하는 데 사용할 수 있는 사용자 지정 도구를 만들어 보겠습니다.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

`langchain_community.retrievers` 모듈의 `BM25Retriever`를 사용하여 리트리버 도구를 만듭니다.

<Tip>
  <code>BM25Retriever</code>는 검색을 위한 훌륭한 출발점이지만, 더 고급 의미 검색을 위해서는 <a href="https://www.sbert.net/">sentence-transformers</a>와 같은 임베딩 기반 리트리버를 사용하는 것을 고려할 수 있습니다.
</Tip>

```python
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever

class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "이름이나 관계를 기반으로 갈라 손님에 대한 자세한 정보를 검색합니다."
    inputs = {
        "query": {
            "type": "string",
            "description": "정보를 원하는 손님의 이름이나 관계입니다."
        }
    }
    output_type = "string"

    def __init__(self, docs):
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.get_relevant_documents(query)
        if results:
            return "

".join([doc.page_content for doc in results[:3]])
        else:
            return "일치하는 손님 정보가 없습니다."

# 도구 초기화
guest_info_tool = GuestInfoRetrieverTool(docs)
```

이 도구를 단계별로 이해해 보겠습니다.
- `name` 및 `description`은 에이전트가 이 도구를 언제 어떻게 사용해야 하는지 이해하는 데 도움이 됩니다.
- `inputs`는 도구가 예상하는 매개변수를 정의합니다(이 경우 검색 쿼리).
- 임베딩이 필요 없는 강력한 텍스트 검색 알고리즘인 `BM25Retriever`를 사용하고 있습니다.
- `forward` 메서드는 쿼리를 처리하고 가장 관련성 높은 손님 정보를 반환합니다.

</hfoption>
<hfoption id="llama-index">

`llama_index.retrievers.bm25` 모듈의 `BM25Retriever`를 사용하여 리트리버 도구를 만듭니다.

<Tip>
  <code>BM25Retriever</code>는 검색을 위한 훌륭한 출발점이지만, 더 고급 의미 검색을 위해서는 <a href="https://www.sbert.net/">sentence-transformers</a>와 같은 임베딩 기반 리트리버를 사용하는 것을 고려할 수 있습니다.
</Tip>

```python
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever

bm25_retriever = BM25Retriever.from_defaults(nodes=docs)

def get_guest_info_retriever(query: str) -> str:
    """이름이나 관계를 기반으로 갈라 손님에 대한 자세한 정보를 검색합니다."""
    results = bm25_retriever.retrieve(query)
    if results:
        return "

".join([doc.text for doc in results[:3]])
    else:
        return "일치하는 손님 정보가 없습니다."

# 도구 초기화
guest_info_tool = FunctionTool.from_defaults(get_guest_info_retriever)
```

이 도구를 단계별로 이해해 보겠습니다.
- docstring은 에이전트가 이 도구를 언제 어떻게 사용해야 하는지 이해하는 데 도움이 됩니다.
- 유형 데코레이터는 도구가 예상하는 매개변수를 정의합니다(이 경우 검색 쿼리).
- 임베딩이 필요 없는 강력한 텍스트 검색 알고리즘인 `BM25Retriever`를 사용하고 있습니다.
- 메서드는 쿼리를 처리하고 가장 관련성 높은 손님 정보를 반환합니다.

</hfoption>
<hfoption id="langgraph">

`langchain_community.retrievers` 모듈의 `BM25Retriever`를 사용하여 리트리버 도구를 만듭니다.

<Tip>
  <code>BM25Retriever</code>는 검색을 위한 훌륭한 출발점이지만, 더 고급 의미 검색을 위해서는 <a href="https://www.sbert.net/">sentence-transformers</a>와 같은 임베딩 기반 리트리버를 사용하는 것을 고려할 수 있습니다.
</Tip>

```python
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool

bm25_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str) -> str:
    """이름이나 관계를 기반으로 갈라 손님에 대한 자세한 정보를 검색합니다."""
    results = bm25_retriever.invoke(query)
    if results:
        return "

".join([doc.page_content for doc in results[:3]])
    else:
        return "일치하는 손님 정보가 없습니다."

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="이름이나 관계를 기반으로 갈라 손님에 대한 자세한 정보를 검색합니다."
)
```

이 도구를 단계별로 이해해 보겠습니다.
- `name` 및 `description`은 에이전트가 이 도구를 언제 어떻게 사용해야 하는지 이해하는 데 도움이 됩니다.
- 유형 데코레이터는 도구가 예상하는 매개변수를 정의합니다(이 경우 검색 쿼리).
- 임베딩이 필요 없는 강력한 텍스트 검색 알고리즘인 `BM25Retriever`를 사용하고 있습니다.
- 메서드는 쿼리를 처리하고 가장 관련성 높은 손님 정보를 반환합니다.


</hfoption>
</hfoptions>

### 3단계: 알프레드와 도구 통합

마지막으로, 에이전트를 만들고 사용자 지정 도구를 장착하여 모든 것을 하나로 모으겠습니다.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
from smolagents import CodeAgent, InferenceClientModel

# Hugging Face 모델 초기화
model = InferenceClientModel()

# 갈라 에이전트인 알프레드를 게스트 정보 도구로 만들기
alfred = CodeAgent(tools=[guest_info_tool], model=model)

# 갈라 동안 알프레드가 받을 수 있는 예시 쿼리
response = alfred.run("'레이디 에이다 러브레이스'라는 이름의 손님에 대해 알려주세요.")

print("🎩 알프레드의 응답:")
print(response)
```

예상 출력:

```
🎩 알프레드의 응답:
제가 검색한 정보에 따르면, 레이디 에이다 러브레이스는 존경받는 수학자이자 친구입니다. 그녀는 찰스 배비지의 분석 엔진에 대한 연구로 최초의 컴퓨터 프로그래머로 종종 칭송받는 수학 및 컴퓨팅 분야의 선구적인 업적으로 유명합니다. 그녀의 이메일 주소는 ada.lovelace@example.com입니다.
```

이 마지막 단계에서 일어나는 일:
- `InferenceClientModel` 클래스를 사용하여 Hugging Face 모델을 초기화합니다.
- 문제를 해결하기 위해 Python 코드를 실행할 수 있는 `CodeAgent`로 에이전트(알프레드)를 만듭니다.
- 알프레드에게 "레이디 에이다 러브레이스"라는 이름의 손님에 대한 정보를 검색하도록 요청합니다.

</hfoption>
<hfoption id="llama-index">

```python
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# Hugging Face 모델 초기화
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# 갈라 에이전트인 알프레드를 게스트 정보 도구로 만들기
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool],
    llm=llm,
)

# 갈라 동안 알프레드가 받을 수 있는 예시 쿼리
response = await alfred.run("'레이디 에이다 러브레이스'라는 이름의 손님에 대해 알려주세요.")

print("🎩 알프레드의 응답:")
print(response)
```

예상 출력:

```
🎩 알프레드의 응답:
레이디 에이다 러브레이스는 존경받는 수학자이자 친구이며, 수학 및 컴퓨팅 분야의 선구적인 업적으로 유명합니다. 그녀는 찰스 배비지의 분석 엔진에 대한 연구로 최초의 컴퓨터 프로그래머로 칭송받습니다. 그녀의 이메일은 ada.lovelace@example.com입니다.
```

이 마지막 단계에서 일어나는 일:
- `HuggingFaceInferenceAPI` 클래스를 사용하여 Hugging Face 모델을 초기화합니다.
- 방금 만든 도구를 포함하여 `AgentWorkflow`로 에이전트(알프레드)를 만듭니다.
- 알프레드에게 "레이디 에이다 러브레이스"라는 이름의 손님에 대한 정보를 검색하도록 요청합니다.

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
tools = [guest_info_tool]
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

messages = [HumanMessage(content="'레이디 에이다 러브레이스'라는 이름의 손님에 대해 알려주세요.")]
response = alfred.invoke({"messages": messages})

print("🎩 알프레드의 응답:")
print(response['messages'][-1].content)
```

예상 출력:

```
🎩 알프레드의 응답:
레이디 에이다 러브레이스는 존경받는 수학자이자 컴퓨팅의 선구자로, 찰스 배비지의 분석 엔진에 대한 연구로 최초의 컴퓨터 프로그래머로 종종 칭송받습니다.
```

이 마지막 단계에서 일어나는 일:
- `HuggingFaceEndpoint` 클래스를 사용하여 Hugging Face 모델을 초기화합니다. 채팅 인터페이스를 생성하고 도구를 추가합니다.
- 2개의 노드(`assistant`, `tools`)를 엣지로 결합하는 `StateGraph`로 에이전트(알프레드)를 만듭니다.
- 알프레드에게 "레이디 에이다 러브레이스"라는 이름의 손님에 대한 정보를 검색하도록 요청합니다.

</hfoption>
</hfoptions>

## 예시 상호 작용

갈라 동안 대화는 다음과 같이 흐를 수 있습니다.

**당신:** "알프레드, 대사와 이야기하는 저 신사는 누구입니까?"

**알프레드:** *신속하게 손님 데이터베이스를 검색합니다* "저분은 니콜라 테슬라 박사님입니다. 대학 시절의 오랜 친구이십니다. 최근에 새로운 무선 에너지 전송 시스템에 대한 특허를 받았으며 이에 대해 기꺼이 논의할 것입니다. 비둘기에 대한 열정이 있다는 것을 기억하십시오. 그래서 그것이 좋은 잡담거리가 될 수 있습니다."

```json
{
    "name": "니콜라 테슬라 박사",
    "relation": "대학 시절의 오랜 친구",
    "description": "니콜라 테슬라 박사는 대학 시절의 오랜 친구입니다. 최근에 새로운 무선 에너지 전송 시스템에 대한 특허를 받았으며 이에 대해 기꺼이 논의할 것입니다. 비둘기에 대한 열정이 있다는 것을 기억하십시오. 그래서 그것이 좋은 잡담거리가 될 수 있습니다.",
    "email": "nikola.tesla@gmail.com"
}
```

## 더 나아가기

이제 알프레드가 손님 정보를 검색할 수 있으므로 이 시스템을 어떻게 향상시킬 수 있는지 고려해 보십시오.

1. [sentence-transformers](https://www.sbert.net/)와 같은 더 정교한 알고리즘을 사용하도록 리트리버를 개선합니다.
2. 알프레드가 이전 상호 작용을 기억하도록 대화 메모리를 구현합니다.
3. 익숙하지 않은 손님에 대한 최신 정보를 얻기 위해 웹 검색과 결합합니다.
4. 검증된 출처에서 더 완전한 정보를 얻기 위해 여러 인덱스를 통합합니다.

이제 알프레드는 손님 문의를 손쉽게 처리할 수 있도록 완벽하게 갖추어져 있어 갈라가 세기의 가장 정교하고 즐거운 행사로 기억되도록 보장합니다!

<Tip>
각 손님의 관심사나 배경을 바탕으로 대화 시작점을 반환하는 리트리버 도구를 확장해 보십시오. 이 작업을 수행하기 위해 도구를 어떻게 수정하시겠습니까?

완료되면 <code>retriever.py</code> 파일에 게스트 리트리버 도구를 구현하십시오.
</Tip>