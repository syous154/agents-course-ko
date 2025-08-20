# LlamaIndex에서 에이전트 워크플로우 생성하기

LlamaIndex의 워크플로우는 코드를 순차적이고 관리 가능한 단계로 구성하는 구조화된 방법을 제공합니다.

이러한 워크플로우는 `Events`에 의해 트리거되고, 스스로 `Events`를 방출하여 추가 단계를 트리거하는 `Steps`를 정의함으로써 생성됩니다.
Alfred가 RAG 작업을 위한 LlamaIndex 워크플로우를 보여주는 것을 살펴보겠습니다.

![Workflow Schematic](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/workflows.png)

**워크플로우는 몇 가지 주요 이점을 제공합니다:**

- 코드를 개별 단계로 명확하게 구성
- 유연한 제어 흐름을 위한 이벤트 기반 아키텍처
- 단계 간 타입 안전 통신
- 내장된 상태 관리
- 간단하고 복잡한 에이전트 상호 작용 모두 지원

짐작하셨겠지만, **워크플로우는 에이전트의 자율성과 전체 워크플로우에 대한 제어 사이에서 훌륭한 균형을 이룹니다.**

그럼, 직접 워크플로우를 만들어 봅시다!

## 워크플로우 생성하기

<Tip>
<a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/workflows.ipynb" target="_blank">이 노트북</a>의 코드를 따라 할 수 있으며, Google Colab을 사용하여 실행할 수 있습니다.
</Tip>

### 기본 워크플로우 생성

<details>
<summary>워크플로우 패키지 설치</summary>
<a href="./llama-hub">LlamaHub 섹션</a>에서 소개된 바와 같이, 다음 명령어를 사용하여 워크플로우 패키지를 설치할 수 있습니다.

```python
pip install llama-index-utils-workflow
```
</details>

`Workflow`를 상속받는 클래스를 정의하고 `@step` 데코레이터로 함수를 장식하여 단일 단계 워크플로우를 생성할 수 있습니다.
또한 워크플로우의 시작과 끝을 나타내는 데 사용되는 특수 이벤트인 `StartEvent`와 `StopEvent`를 추가해야 합니다.

```python
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")


w = MyWorkflow(timeout=10, verbose=False)
result = await w.run()
```

보시다시피, 이제 `w.run()`을 호출하여 워크플로우를 실행할 수 있습니다.

### 여러 단계 연결하기

여러 단계를 연결하려면 **단계 간에 데이터를 전달하는 사용자 지정 이벤트를 생성해야 합니다.**
이를 위해 단계 간에 전달되어 첫 번째 단계의 출력을 두 번째 단계로 전달하는 `Event`를 추가해야 합니다.

```python
from llama_index.core.workflow import Event

class ProcessingEvent(Event):
    intermediate_result: str

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        # Process initial data
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

w = MultiStepWorkflow(timeout=10, verbose=False)
result = await w.run()
result
```

여기서 타입 힌팅은 워크플로우가 올바르게 실행되도록 보장하므로 중요합니다. 이제 좀 더 복잡하게 만들어 봅시다!

### 루프 및 분기

타입 힌팅은 워크플로우의 가장 강력한 부분입니다. 이를 통해 분기, 루프 및 조인을 생성하여 더 복잡한 워크플로우를 용이하게 할 수 있습니다.

합집합 연산자 `|`를 사용하여 **루프를 생성하는 예시**를 보여드리겠습니다.
아래 예시에서 `LoopEvent`는 단계의 입력으로 사용될 수 있으며 출력으로도 반환될 수 있음을 알 수 있습니다.

```python
from llama_index.core.workflow import Event
import random


class ProcessingEvent(Event):
    intermediate_result: str


class LoopEvent(Event):
    loop_output: str


class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


w = MultiStepWorkflow(verbose=False)
result = await w.run()
result
```

### 워크플로우 그리기

워크플로우를 그릴 수도 있습니다. `draw_all_possible_flows` 함수를 사용하여 워크플로우를 그려봅시다. 이 함수는 워크플로우를 HTML 파일로 저장합니다.

```python
from llama_index.utils.workflow import draw_all_possible_flows

w = ... # as defined in the previous section
draw_all_possible_flows(w, "flow.html")
```

![workflow drawing](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/workflow-draw.png)

과정에서 다룰 마지막 멋진 트릭은 워크플로우에 상태를 추가하는 기능입니다.

### 상태 관리

상태 관리는 워크플로우의 상태를 추적하여 모든 단계가 동일한 상태에 접근할 수 있도록 할 때 유용합니다.
이는 단계 함수에서 매개변수 위에 `Context` 타입 힌트를 사용하여 수행할 수 있습니다.

```python
from llama_index.core.workflow import Context, StartEvent, StopEvent


@step
async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
    # store query in the context
    await ctx.store.set("query", "What is the capital of France?")

    # do something with context and event
    val = ...

    # retrieve query from the context
    query = await ctx.store.get("query")

    return StopEvent(result=val)
```

훌륭합니다! 이제 LlamaIndex에서 기본 워크플로우를 생성하는 방법을 알게 되었습니다!

<Tip>워크플로우에는 더 복잡한 뉘앙스가 있으며, <a href="https://docs.llamaindex.ai/en/stable/understanding/workflows/">LlamaIndex 문서</a>에서 자세히 알아볼 수 있습니다.</Tip>

하지만 워크플로우를 생성하는 또 다른 방법이 있는데, 이는 `AgentWorkflow` 클래스에 의존합니다. 이 클래스를 사용하여 다중 에이전트 워크플로우를 생성하는 방법을 살펴보겠습니다.

## 다중 에이전트 워크플로우로 워크플로우 자동화하기

수동 워크플로우 생성 대신, **`AgentWorkflow` 클래스를 사용하여 다중 에이전트 워크플로우를 생성할 수 있습니다.**
`AgentWorkflow`는 워크플로우 에이전트를 사용하여 전문화된 기능을 기반으로 서로 협력하고 작업을 인계할 수 있는 하나 이상의 에이전트 시스템을 생성할 수 있도록 합니다.
이를 통해 다양한 에이전트가 작업의 다른 측면을 처리하는 복잡한 에이전트 시스템을 구축할 수 있습니다.
`llama_index.core.agent`에서 클래스를 가져오는 대신, `llama_index.core.agent.workflow`에서 에이전트 클래스를 가져올 것입니다.
`AgentWorkflow` 생성자에서 하나의 에이전트를 루트 에이전트로 지정해야 합니다.
사용자 메시지가 들어오면 먼저 루트 에이전트로 라우팅됩니다.

각 에이전트는 다음을 수행할 수 있습니다.

- 도구를 사용하여 요청을 직접 처리
- 작업에 더 적합한 다른 에이전트에게 인계
- 사용자에게 응답 반환

다중 에이전트 워크플로우를 생성하는 방법을 살펴보겠습니다.

```python
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# we can pass functions directly without FunctionTool -- the fn/docstring are parsed for the name/description
multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Is able to multiply two integers",
    system_prompt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply],
    llm=llm,
)

addition_agent = ReActAgent(
    name="add_agent",
    description="Is able to add two integers",
    system_prompt="A helpful assistant that can use a tool to add numbers.",
    tools=[add],
    llm=llm,
)

# Create the workflow
workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
)

# Run the system
response = await workflow.run(user_msg="Can you add 5 and 3?")
```

에이전트 도구는 앞서 언급한 워크플로우 상태를 수정할 수도 있습니다. 워크플로우를 시작하기 전에 모든 에이전트가 사용할 수 있는 초기 상태 딕셔너리를 제공할 수 있습니다.
상태는 워크플로우 컨텍스트의 상태 키에 저장됩니다. 이는 각 새 사용자 메시지를 증강하는 `state_prompt`에 주입됩니다.

이전 예시를 수정하여 함수 호출 횟수를 세는 카운터를 주입해 봅시다.

```python
from llama_index.core.workflow import Context

# Define some tools
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    # update our count
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)

    return a * b

async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    # update our count
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)

    return a * b

...

workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
    initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}",
)

# run the workflow with context
ctx = Context(workflow)
response = await workflow.run(user_msg="Can you add 5 and 3?")

# pull out and inspect the state
state = await ctx.store.get("state")
print(state["num_fn_calls"])
```

축하합니다! 이제 LlamaIndex의 에이전트 기본 사항을 마스터했습니다! 🎉

지식을 굳건히 하기 위해 마지막 퀴즈를 풀어봅시다! 🚀