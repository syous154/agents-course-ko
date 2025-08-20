# LlamaIndex에서 도구 사용하기

**명확한 도구 세트를 정의하는 것이 성능에 중요합니다.** [1단원](../../unit1/tools)에서 논의했듯이, 명확한 도구 인터페이스는 LLM이 사용하기 더 쉽습니다. 인간 엔지니어를 위한 소프트웨어 API 인터페이스와 마찬가지로, 도구의 작동 방식을 이해하기 쉬우면 도구를 더 잘 활용할 수 있습니다.

LlamaIndex에는 **네 가지 주요 도구 유형**이 있습니다.

![Tools](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/tools.png)

1.  `FunctionTool`: 모든 Python 함수를 에이전트가 사용할 수 있는 도구로 변환합니다. 함수의 작동 방식을 자동으로 파악합니다.
2.  `QueryEngineTool`: 에이전트가 쿼리 엔진을 사용할 수 있도록 하는 도구입니다. 에이전트는 쿼리 엔진을 기반으로 구축되므로 다른 에이전트를 도구로 사용할 수도 있습니다.
3.  `Toolspecs`: 커뮤니티에서 생성한 도구 세트로, Gmail과 같은 특정 서비스를 위한 도구가 포함되는 경우가 많습니다.
4.  `Utility Tools`: 다른 도구에서 가져온 대량의 데이터를 처리하는 데 도움이 되는 특수 도구입니다.

각각에 대해 아래에서 더 자세히 살펴보겠습니다.

## FunctionTool 생성하기

<Tip>
Google Colab을 사용하여 실행할 수 있는 <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/tools.ipynb" target="_blank">이 노트북</a>의 코드를 따라 할 수 있습니다.
</Tip>

FunctionTool은 모든 Python 함수를 래핑하고 에이전트가 사용할 수 있도록 하는 간단한 방법을 제공합니다. 도구에 동기 또는 비동기 함수를 선택적 `name` 및 `description` 매개변수와 함께 전달할 수 있습니다. 이름과 설명은 에이전트가 도구를 언제 어떻게 효과적으로 사용해야 하는지 이해하는 데 도움이 되므로 특히 중요합니다. 아래에서 FunctionTool을 생성하고 호출하는 방법을 살펴보겠습니다.

```python
from llama_index.core.tools import FunctionTool

def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"

tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location.",
)
tool.call("New York")
```

<Tip>에이전트 또는 LLM을 함수 호출과 함께 사용할 때, 선택된 도구(및 해당 도구에 대해 작성된 인수)는 도구의 이름과 도구의 목적 및 인수에 대한 설명에 크게 의존합니다. 함수 호출에 대한 자세한 내용은 <a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">함수 호출 가이드</a>에서 확인할 수 있습니다.</Tip>

## QueryEngineTool 생성하기

이전 단원에서 정의한 `QueryEngine`은 `QueryEngineTool` 클래스를 사용하여 쉽게 도구로 변환할 수 있습니다. 아래 예제에서 `QueryEngine`에서 `QueryEngineTool`을 생성하는 방법을 살펴보겠습니다.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(llm=llm)
tool = QueryEngineTool.from_defaults(query_engine, name="some useful name", description="some useful description")
```

## Toolspecs 생성하기

`ToolSpecs`를 조화롭게 함께 작동하는 도구 모음, 즉 잘 정리된 전문 도구 키트라고 생각해보세요. 정비공의 도구 키트에 차량 수리를 위해 함께 작동하는 보완 도구가 포함되어 있듯이, `ToolSpec`은 특정 목적을 위해 관련 도구를 결합합니다. 예를 들어, 회계 에이전트의 `ToolSpec`은 스프레드시트 기능, 이메일 기능 및 계산 도구를 우아하게 통합하여 재무 작업을 정밀하고 효율적으로 처리할 수 있습니다.

<details>
<summary>Google Toolspec 설치</summary>
<a href="./llama-hub">LlamaHub 섹션</a>에서 소개했듯이, 다음 명령으로 Google toolspec을 설치할 수 있습니다.

```python
pip install llama-index-tools-google
```
</details>

이제 toolspec을 로드하고 도구 목록으로 변환할 수 있습니다.

```python
from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()
```

도구에 대한 더 자세한 정보를 얻으려면 각 도구의 `metadata`를 살펴볼 수 있습니다.

```python
[(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list]
```

### LlamaIndex의 모델 컨텍스트 프로토콜 (MCP)

LlamaIndex는 [LlamaHub의 ToolSpec](https://llamahub.ai/l/tools/llama-index-tools-mcp?from=)을 통해 MCP 도구를 사용하는 것도 허용합니다. MCP 서버를 실행하고 다음 구현을 통해 사용을 시작할 수 있습니다.

MCP에 대해 더 자세히 알아보고 싶다면 [무료 MCP 강좌](https://huggingface.co/learn/mcp-course/)를 확인하세요.

<details>
<summary>MCP Toolspec 설치</summary>
<a href="./llama-hub">LlamaHub 섹션</a>에서 소개했듯이, 다음 명령으로 MCP toolspec을 설치할 수 있습니다.

```python
pip install llama-index-tools-mcp
```
</details>

```python
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# We consider there is a mcp server running on 127.0.0.1:8000, or you can use the mcp client to connect to your own mcp server.
mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
mcp_tool = McpToolSpec(client=mcp_client)

# get the agent
agent = await get_agent(mcp_tool)

# create the agent context
agent_context = Context(agent)
```

## 유틸리티 도구

종종 API를 직접 쿼리하면 **과도한 양의 데이터가 반환될 수 있으며**, 이 중 일부는 관련이 없거나 LLM의 컨텍스트 창을 넘치게 하거나 사용 중인 토큰 수를 불필요하게 증가시킬 수 있습니다. 아래에서 두 가지 주요 유틸리티 도구를 살펴보겠습니다.

1.  `OnDemandToolLoader`: 이 도구는 기존 LlamaIndex 데이터 로더(BaseReader 클래스)를 에이전트가 사용할 수 있는 도구로 변환합니다. 이 도구는 데이터 로더에서 `load_data`를 트리거하는 데 필요한 모든 매개변수와 자연어 쿼리 문자열과 함께 호출될 수 있습니다. 실행 중에 먼저 데이터 로더에서 데이터를 로드하고, 이를 인덱싱(예: 벡터 저장소 사용)한 다음 '온디맨드'로 쿼리합니다. 이 세 가지 단계는 모두 단일 도구 호출에서 발생합니다.
2.  `LoadAndSearchToolSpec`: LoadAndSearchToolSpec은 기존 도구를 입력으로 받습니다. 도구 사양으로서 `to_tool_list`를 구현하며, 이 함수가 호출되면 로딩 도구와 검색 도구의 두 가지 도구가 반환됩니다. 로드 도구 실행은 기본 도구를 호출한 다음 출력을 인덱싱합니다(기본적으로 벡터 인덱스 사용). 검색 도구 실행은 쿼리 문자열을 입력으로 받아 기본 인덱스를 호출합니다.

<Tip><a href="https://llamahub.ai/">LlamaHub</a>에서 toolspec과 유틸리티 도구를 찾을 수 있습니다.</Tip>

이제 LlamaIndex의 에이전트와 도구의 기본 사항을 이해했으므로, **LlamaIndex를 사용하여 구성 가능하고 관리 가능한 워크플로를 생성하는 방법**을 살펴보겠습니다!