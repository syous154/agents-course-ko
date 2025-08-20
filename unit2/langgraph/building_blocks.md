# LangGraph의 구성 요소

LangGraph로 애플리케이션을 구축하려면 핵심 구성 요소를 이해해야 합니다. LangGraph 애플리케이션을 구성하는 기본적인 빌딩 블록을 살펴보겠습니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/Building_blocks.png" alt="Building Blocks" width="70%"/>

LangGraph의 애플리케이션은 **진입점(entrypoint)**에서 시작하며, 실행에 따라 흐름은 END에 도달할 때까지 한 함수에서 다른 함수로 이동할 수 있습니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/application.png" alt="Application"/>

## 1. 상태 (State)

**상태(State)**는 LangGraph의 핵심 개념입니다. 이는 애플리케이션을 통해 흐르는 모든 정보를 나타냅니다.

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

상태는 **사용자 정의**이므로, 의사 결정 프로세스에 필요한 모든 데이터를 포함하도록 필드를 신중하게 작성해야 합니다!

> 💡 **팁:** 애플리케이션이 단계별로 추적해야 하는 정보가 무엇인지 신중하게 고려하세요.

## 2. 노드 (Nodes)

**노드(Nodes)**는 파이썬 함수입니다. 각 노드는 다음을 수행합니다.
- 상태를 입력으로 받습니다.
- 일부 작업을 수행합니다.
- 상태에 대한 업데이트를 반환합니다.

```python
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}
```

예를 들어, 노드에는 다음이 포함될 수 있습니다.
- **LLM 호출**: 텍스트를 생성하거나 결정을 내립니다.
- **도구 호출**: 외부 시스템과 상호 작용합니다.
- **조건부 로직**: 다음 단계를 결정합니다.
- **인간 개입**: 사용자로부터 입력을 받습니다.

> 💡 **정보:** START 및 END와 같이 전체 워크플로우에 필요한 일부 노드는 LangGraph에서 직접 제공됩니다.

## 3. 엣지 (Edges)

**엣지(Edges)**는 노드를 연결하고 그래프를 통한 가능한 경로를 정의합니다.

```python
import random
from typing import Literal

def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"
```

엣지는 다음 중 하나일 수 있습니다.
- **직접**: 항상 노드 A에서 노드 B로 이동합니다.
- **조건부**: 현재 상태를 기반으로 다음 노드를 선택합니다.

## 4. StateGraph

**StateGraph**는 전체 에이전트 워크플로우를 담는 컨테이너입니다.

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()
```

그런 다음 시각화할 수 있습니다!
```python
# View
display(Image(graph.get_graph().draw_mermaid_png()))
```
<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/basic_graph.jpeg" alt="Graph Visualization"/>

하지만 가장 중요하게는 호출할 수 있습니다.
```python
graph.invoke({"graph_state" : "Hi, this is Lance."})
```
출력:
```
---Node 1---
---Node 3---
{'graph_state': 'Hi, this is Lance. I am sad!'}
```

## 다음 단계는?

다음 섹션에서는 이러한 개념을 실제로 적용하여 첫 번째 그래프를 구축할 것입니다. 이 그래프를 통해 Alfred는 이메일을 받아 분류하고, 이메일이 진짜인 경우 예비 답변을 작성할 수 있습니다.