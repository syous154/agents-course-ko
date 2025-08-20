# LlamaIndex에서 에이전트 사용하기

이전에 보았던 우리의 유용한 집사 에이전트 알프레드를 기억하시나요? 이제 그가 업그레이드될 예정입니다! LlamaIndex에서 사용할 수 있는 도구들을 이해했으니, 알프레드에게 새로운 기능을 부여하여 우리를 더 잘 돕도록 할 수 있습니다.

하지만 계속하기 전에, 알프레드와 같은 에이전트를 움직이게 하는 것이 무엇인지 다시 한번 상기해 봅시다. 1단원에서 우리는 다음을 배웠습니다.

> 에이전트는 AI 모델을 활용하여 환경과 상호 작용하며 사용자 정의 목표를 달성하는 시스템입니다. 이는 추론, 계획, 그리고 행동 실행(종종 외부 도구를 통해)을 결합하여 작업을 수행합니다.

LlamaIndex는 **세 가지 주요 추론 에이전트 유형**을 지원합니다.

![Agents](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/agents.png)

1.  `Function Calling Agents` - 이들은 특정 함수를 호출할 수 있는 AI 모델과 함께 작동합니다.
2.  `ReAct Agents` - 이들은 채팅 또는 텍스트 엔드포인트를 처리하고 복잡한 추론 작업을 다루는 모든 AI와 함께 작동할 수 있습니다.
3.  `Advanced Custom Agents` - 이들은 더 복잡한 작업과 워크플로우를 처리하기 위해 더 복잡한 방법을 사용합니다.

<Tip>고급 에이전트에 대한 자세한 정보는 <a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/agent/workflow/base_agent.py">BaseWorkflowAgent</a>에서 찾을 수 있습니다.</Tip>

## 에이전트 초기화

<Tip>
<a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/agents.ipynb" target="_blank">이 노트북</a>의 코드를 따라 할 수 있으며, Google Colab을 사용하여 실행할 수 있습니다.
</Tip>

에이전트를 생성하려면 먼저 **에이전트의 기능을 정의하는 함수/도구 세트**를 제공해야 합니다. 몇 가지 기본 도구를 사용하여 에이전트를 생성하는 방법을 살펴보겠습니다. 이 글을 쓰는 시점에는 에이전트가 자동으로 함수 호출 API(사용 가능한 경우) 또는 표준 ReAct 에이전트 루프를 사용합니다.

도구/함수 API를 지원하는 LLM은 비교적 새로운 기술이지만, 특정 프롬프트를 피하고 LLM이 제공된 스키마를 기반으로 도구 호출을 생성하도록 허용함으로써 도구를 호출하는 강력한 방법을 제공합니다.

ReAct 에이전트는 복잡한 추론 작업에도 능숙하며 채팅 또는 텍스트 완성 기능을 가진 모든 LLM과 함께 작동할 수 있습니다. 이들은 더 장황하며, 수행하는 특정 행동 뒤에 있는 추론을 보여줍니다.

```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

# initialize llm
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# initialize agent
agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)
```

**에이전트는 기본적으로 상태 비저장(stateless)입니다.** 그러나 `Context` 객체를 사용하여 과거 상호 작용을 기억할 수 있습니다. 이는 여러 메시지에서 컨텍스트를 유지하는 챗봇이나 시간 경과에 따른 진행 상황을 추적해야 하는 작업 관리자와 같이 이전 상호 작용을 기억해야 하는 에이전트를 사용하려는 경우 유용할 수 있습니다.

```python
# stateless
response = await agent.run("What is 2 times 2?")

# remembering state
from llama_index.core.workflow import Context

ctx = Context(agent)

response = await agent.run("My name is Bob.", ctx=ctx)
response = await agent.run("What was my name again?", ctx=ctx)
```

LlamaIndex의 에이전트는 Python의 `await` 연산자를 사용하기 때문에 비동기(async)라는 것을 알 수 있습니다. Python의 비동기 코드에 익숙하지 않거나 복습이 필요한 경우, [훌륭한 비동기 가이드](https://docs.llamaindex.ai/en/stable/getting_started/async_python/)가 있습니다.

이제 기본 사항을 알았으니, 에이전트에서 더 복잡한 도구를 사용하는 방법을 살펴보겠습니다.

## QueryEngineTools로 RAG 에이전트 생성하기

**에이전트 RAG는 에이전트를 사용하여 데이터에 대한 질문에 답변하는 강력한 방법입니다.** 알프레드가 질문에 답변하는 데 도움이 되도록 다양한 도구를 전달할 수 있습니다. 그러나 알프레드는 문서 위에 자동으로 질문에 답변하는 대신, 질문에 답변하기 위해 다른 도구나 흐름을 사용할 수 있습니다.

![Agentic RAG](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/agentic-rag.png)

`QueryEngine`을 에이전트의 도구로 **래핑하는 것은 쉽습니다.** 이 작업을 수행할 때 **이름과 설명을 정의**해야 합니다. LLM은 이 정보를 사용하여 도구를 올바르게 사용합니다. [구성 요소 섹션](components)에서 생성한 `QueryEngine`을 사용하여 `QueryEngineTool`을 로드하는 방법을 살펴보겠습니다.

```python
from llama_index.core.tools import QueryEngineTool

query_engine = index.as_query_engine(llm=llm, similarity_top_k=3) # as shown in the Components in LlamaIndex section

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="name",
    description="a specific description",
    return_direct=False,
)
query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. "
)
```

## 다중 에이전트 시스템 생성하기

`AgentWorkflow` 클래스는 다중 에이전트 시스템도 직접 지원합니다. 각 에이전트에 이름과 설명을 부여함으로써 시스템은 단일 활성 발화자를 유지하며, 각 에이전트는 다른 에이전트에게 작업을 넘길 수 있습니다. 각 에이전트의 범위를 좁힘으로써 사용자 메시지에 응답할 때 전반적인 정확도를 높일 수 있습니다.

LlamaIndex의 에이전트는 더 복잡하고 사용자 정의된 시나리오를 위해 **다른 에이전트의 도구로 직접 사용될 수도 있습니다.**

```python
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm
)

# Create and run the workflow
agent = AgentWorkflow(
    agents=[calculator_agent, query_agent], root_agent="calculator"
)

# Run the system
response = await agent.run(user_msg="Can you add 5 and 3?")
```

<Tip>아직 충분히 배우지 못했나요? 스트리밍, 컨텍스트 직렬화, 휴먼 인 더 루프에 대해 더 자세히 알아볼 수 있는 <a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/">AgentWorkflow 기본 소개</a> 또는 <a href="https://docs.llamaindex.ai/en/stable/understanding/agent/">에이전트 학습 가이드</a>에서 LlamaIndex의 에이전트와 도구에 대해 더 많은 것을 발견할 수 있습니다!</Tip>

이제 LlamaIndex의 에이전트와 도구의 기본 사항을 이해했으니, LlamaIndex를 사용하여 **구성 가능하고 관리 가능한 워크플로우를 생성하는 방법**을 살펴보겠습니다!