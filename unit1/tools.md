# 도구란 무엇인가요?

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/whiteboard-check-2.jpg" alt="1단원 계획"/>

AI 에이전트의 중요한 측면 중 하나는 **행동**을 취할 수 있는 능력입니다. 우리가 보았듯이, 이는 **도구**의 사용을 통해 이루어집니다.

이 섹션에서는 도구가 무엇인지, 도구를 효과적으로 설계하는 방법, 그리고 시스템 메시지를 통해 에이전트에 도구를 통합하는 방법을 배울 것입니다.

에이전트에 적절한 도구를 제공하고 해당 도구가 어떻게 작동하는지 명확하게 설명함으로써, AI가 달성할 수 있는 것을 극적으로 늘릴 수 있습니다. 자세히 알아보겠습니다!

## AI 도구란 무엇인가요?

**도구는 LLM에 제공되는 함수**입니다. 이 함수는 **명확한 목표**를 달성해야 합니다.

다음은 AI 에이전트에서 일반적으로 사용되는 도구들입니다.

| 도구            | 설명                                                   |
|----------------|---------------------------------------------------------------|
| 웹 검색     | 에이전트가 인터넷에서 최신 정보를 가져올 수 있도록 합니다. |
| 이미지 생성 | 텍스트 설명을 기반으로 이미지를 생성합니다.                  |
| 검색      | 외부 소스에서 정보를 검색합니다.                |
| API 인터페이스  | 외부 API(GitHub, YouTube, Spotify 등)와 상호 작용합니다. |

이것들은 단지 예시일 뿐이며, 사실 어떤 사용 사례에 대해서도 도구를 만들 수 있습니다!

좋은 도구는 **LLM의 능력을 보완**할 수 있는 것이어야 합니다.

예를 들어, 산술 연산을 수행해야 하는 경우, LLM에 **계산기 도구**를 제공하면 모델의 기본 기능에 의존하는 것보다 더 나은 결과를 얻을 수 있습니다.

또한, **LLM은 훈련 데이터를 기반으로 프롬프트의 완성을 예측**합니다. 이는 LLM의 내부 지식이 훈련 이전의 이벤트만 포함한다는 것을 의미합니다. 따라서 에이전트에 최신 데이터가 필요한 경우, 어떤 도구를 통해 이를 제공해야 합니다.

예를 들어, LLM에 직접(검색 도구 없이) 오늘의 날씨를 물어보면, LLM은 무작위 날씨를 환각할 수 있습니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/weather.jpg" alt="날씨"/>

- 도구는 다음을 포함해야 합니다.

  - **함수가 무엇을 하는지에 대한 텍스트 설명**.
  - *호출 가능(Callable)* (행동을 수행하는 것).
  - 타입이 지정된 *인수(Arguments)*.
  - (선택 사항) 타입이 지정된 출력(Outputs).

## 도구는 어떻게 작동하나요?

LLM은 우리가 보았듯이 텍스트 입력만 받고 텍스트 출력만 생성할 수 있습니다. 스스로 도구를 호출할 방법이 없습니다. 에이전트에 도구를 제공한다는 것은 LLM에 이러한 도구의 존재를 가르치고 필요할 때 텍스트 기반 호출을 생성하도록 지시하는 것을 의미합니다.

예를 들어, 인터넷에서 특정 위치의 날씨를 확인하는 도구를 제공한 다음 LLM에 파리의 날씨에 대해 물어보면, LLM은 이것이 "날씨" 도구를 사용할 기회임을 인식할 것입니다. LLM은 날씨 데이터를 직접 검색하는 대신, `call weather_tool('Paris')`와 같이 도구 호출을 나타내는 텍스트를 생성할 것입니다.

그러면 **에이전트**는 이 응답을 읽고, 도구 호출이 필요하다는 것을 식별하고, LLM을 대신하여 도구를 실행하고, 실제 날씨 데이터를 검색합니다.

도구 호출 단계는 일반적으로 사용자에게 표시되지 않습니다. 에이전트는 업데이트된 대화를 LLM에 다시 전달하기 전에 이를 새 메시지로 추가합니다. 그러면 LLM은 이 추가 컨텍스트를 처리하고 사용자에게 자연스러운 응답을 생성합니다. 사용자의 관점에서는 LLM이 도구와 직접 상호 작용한 것처럼 보이지만, 실제로는 에이전트가 백그라운드에서 전체 실행 프로세스를 처리한 것입니다.

이 과정에 대해서는 향후 세션에서 더 자세히 이야기할 것입니다.

## LLM에 도구를 어떻게 제공하나요?

완전한 답변은 압도적으로 보일 수 있지만, 본질적으로 시스템 프롬프트를 사용하여 모델에 사용 가능한 도구에 대한 텍스트 설명을 제공합니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/Agent_system_prompt.png" alt="도구를 위한 시스템 프롬프트"/>

이를 위해서는 다음 사항에 대해 매우 정확하고 명확해야 합니다.

1.  **도구가 무엇을 하는지**
2.  **정확히 어떤 입력을 기대하는지**

이것이 도구 설명이 일반적으로 컴퓨터 언어나 JSON과 같이 표현력이 풍부하지만 정확한 구조를 사용하여 제공되는 이유입니다. 반드시 그렇게 할 필요는 없으며, 어떤 정확하고 일관된 형식이라도 작동할 것입니다.

이것이 너무 이론적으로 들린다면, 구체적인 예시를 통해 이해해 봅시다.

두 정수를 곱하는 단순화된 **계산기** 도구를 구현할 것입니다. 다음은 Python 구현입니다.

```python
def calculator(a: int, b: int) -> int:
    """두 정수를 곱합니다."""
    return a * b
```

따라서 우리 도구는 `calculator`라고 불리며, **두 정수를 곱하고**, 다음 입력을 필요로 합니다.

-   **`a`** (*int*): 정수.
-   **`b`** (*int*): 정수.

도구의 출력은 다음과 같이 설명할 수 있는 또 다른 정수입니다.
-   (*int*): `a`와 `b`의 곱.

이 모든 세부 사항이 중요합니다. LLM이 이해할 수 있도록 도구를 설명하는 텍스트 문자열로 이들을 함께 넣어 봅시다.

```text
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

> **참고:** 이 텍스트 설명은 *LLM이 도구에 대해 알아야 할 내용*입니다.

이전 문자열을 LLM에 대한 입력의 일부로 전달하면, 모델은 이를 도구로 인식하고 입력으로 무엇을 전달해야 하는지, 출력으로 무엇을 기대해야 하는지 알게 될 것입니다.

추가 도구를 제공하려면 일관성을 유지하고 항상 동일한 형식을 사용해야 합니다. 이 과정은 취약할 수 있으며, 실수로 일부 세부 사항을 간과할 수 있습니다.

더 나은 방법이 있을까요?

### 도구 섹션 자동 형식 지정

우리 도구는 Python으로 작성되었으며, 구현은 이미 필요한 모든 것을 제공합니다.

-   무엇을 하는지에 대한 설명적인 이름: `calculator`
-   함수의 독스트링 주석으로 제공되는 더 긴 설명: `두 정수를 곱합니다.`
-   입력 및 해당 유형: 함수는 명확하게 두 개의 `int`를 예상합니다.
-   출력의 유형.

사람들이 프로그래밍 언어를 사용하는 이유가 있습니다. 그것들은 표현력이 풍부하고, 간결하며, 정확합니다.

우리는 Python 소스 코드를 LLM을 위한 도구의 *사양*으로 제공할 수 있지만, 도구가 구현되는 방식은 중요하지 않습니다. 중요한 것은 이름, 하는 일, 예상하는 입력 및 제공하는 출력입니다.

Python의 인트로스펙션 기능을 활용하여 소스 코드를 활용하고 도구 설명을 자동으로 구축할 것입니다. 필요한 것은 도구 구현이 타입 힌트, 독스트링, 합리적인 함수 이름을 사용하는 것입니다. 소스 코드에서 관련 부분을 추출할 것입니다.

완료되면, `calculator` 함수가 도구임을 나타내기 위해 Python 데코레이터만 사용하면 됩니다.

```python
@tool
def calculator(a: int, b: int) -> int:
    """두 정수를 곱합니다."""
    return a * b

print(calculator.to_string())
```

함수 정의 앞에 `@tool` 데코레이터가 있음을 주목하십시오.

다음으로 볼 구현을 통해, 데코레이터가 제공하는 `to_string()` 함수를 통해 소스 코드에서 다음 텍스트를 자동으로 검색할 수 있습니다.

```text
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

보시다시피, 이전에 수동으로 작성했던 것과 동일합니다!

### 일반 도구 구현

도구를 사용해야 할 때마다 재사용할 수 있는 일반 `Tool` 클래스를 만듭니다.

> **면책 조항:** 이 예시 구현은 가상의 것이지만, 대부분의 라이브러리에서 실제 구현과 매우 유사합니다.

```python
from typing import Callable


class Tool:
    """
    재사용 가능한 코드 조각(도구)을 나타내는 클래스입니다.

    속성:
        name (str): 도구의 이름.
        description (str): 도구가 무엇을 하는지에 대한 텍스트 설명.
        func (callable): 이 도구가 래핑하는 함수.
        arguments (list): 인수 목록.
        outputs (str 또는 list): 래핑된 함수의 반환 유형.
    """
    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 arguments: list,
                 outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        """
        도구의 이름, 설명, 인수 및 출력을 포함하는 문자열 표현을 반환합니다.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])

        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        """
        제공된 인수로 기본 함수(호출 가능)를 호출합니다.
        """
        return self.func(*args, **kwargs)
```

복잡해 보일 수 있지만, 천천히 살펴보면 무엇을 하는지 알 수 있습니다. 다음을 포함하는 **`Tool`** 클래스를 정의합니다.

-   **`name`** (*str*): 도구의 이름.
-   **`description`** (*str*): 도구가 무엇을 하는지에 대한 간략한 설명.
-   **`function`** (*callable*): 도구가 실행하는 함수.
-   **`arguments`** (*list*): 예상되는 입력 매개변수.
-   **`outputs`** (*str* 또는 *list*): 도구의 예상 출력.
-   **`__call__()`**: 도구 인스턴스가 호출될 때 함수를 호출합니다.
-   **`to_string()`**: 도구의 속성을 텍스트 표현으로 변환합니다.

다음과 같은 코드를 사용하여 이 클래스로 도구를 만들 수 있습니다.

```python
calculator_tool = Tool(
    "calculator",                   # 이름
    "두 정수를 곱합니다.",       # 설명
    calculator,                     # 호출할 함수
    [("a", "int"), ("b", "int")],   # 입력 (이름 및 유형)
    "int",                          # 출력
)
```

하지만 Python의 `inspect` 모듈을 사용하여 모든 정보를 검색할 수도 있습니다! 이것이 `@tool` 데코레이터가 하는 일입니다.

> 관심이 있다면 다음 섹션을 펼쳐 데코레이터 구현을 살펴보세요.

<details>
<summary> 데코레이터 코드</summary>

```python
import inspect

def tool(func):
    """
    주어진 함수에서 Tool 인스턴스를 생성하는 데코레이터입니다.
    """
    # 함수의 시그니처를 가져옵니다.
    signature = inspect.signature(func)

    # 입력에 대한 (매개변수_이름, 매개변수_주석) 쌍을 추출합니다.
    arguments = []
    for param in signature.parameters.values():
        annotation_name = (
            param.annotation.__name__
            if hasattr(param.annotation, '__name__')
            else str(param.annotation)
        )
        arguments.append((param.name, annotation_name))

    # 반환 주석을 결정합니다.
    return_annotation = signature.return_annotation
    if return_annotation is inspect._empty:
        outputs = "반환 주석 없음"
    else:
        outputs = (
            return_annotation.__name__
            if hasattr(return_annotation, '__name__')
            else str(return_annotation)
        )

    # 함수의 독스트링을 설명으로 사용합니다 (기본값: None인 경우).
    description = func.__doc__ or "설명 없음."

    # 함수 이름이 도구 이름이 됩니다.
    name = func.__name__

    # 새 Tool 인스턴스를 반환합니다.
    return Tool(
        name=name,
        description=description,
        func=func,
        arguments=arguments,
        outputs=outputs
    )
```

</details>

다시 한번 강조하지만, 이 데코레이터가 있으면 다음과 같이 도구를 구현할 수 있습니다.

```python
@tool
def calculator(a: int, b: int) -> int:
    """두 정수를 곱합니다."""
    return a * b

print(calculator.to_string())
```

그리고 `Tool`의 `to_string` 메서드를 사용하여 LLM을 위한 도구 설명으로 사용하기에 적합한 텍스트를 자동으로 검색할 수 있습니다.

```text
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

설명은 시스템 프롬프트에 **주입**됩니다. 이 섹션을 시작할 때의 예시를 보면, `tools_description`을 대체한 후 다음과 같이 보일 것입니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/Agent_system_prompt_tools.png" alt="도구를 위한 시스템 프롬프트"/>

[행동](actions) 섹션에서는 에이전트가 방금 만든 이 도구를 **호출**하는 방법에 대해 더 자세히 배울 것입니다.

### 모델 컨텍스트 프로토콜 (MCP): 통합 도구 인터페이스

모델 컨텍스트 프로토콜(MCP)은 애플리케이션이 **LLM에 도구를 제공하는 방법**을 표준화하는 **개방형 프로토콜**입니다.
MCP는 다음을 제공합니다.

-   LLM이 직접 연결할 수 있는 사전 구축된 통합 목록이 증가하고 있습니다.
-   LLM 제공업체 및 공급업체 간에 전환할 수 있는 유연성.
-   인프라 내에서 데이터를 보호하기 위한 모범 사례.

이는 **MCP를 구현하는 모든 프레임워크가 프로토콜 내에 정의된 도구를 활용할 수 있다**는 것을 의미하며, 각 프레임워크에 대해 동일한 인터페이스를 다시 구현할 필요가 없습니다.

MCP에 대해 더 깊이 파고들고 싶다면, [무료 MCP 코스](https://huggingface.co/learn/mcp-course/)를 확인해 보세요.

---

도구는 AI 에이전트의 기능을 향상시키는 데 중요한 역할을 합니다.

요약하자면, 우리는 다음을 배웠습니다.

-   *도구란 무엇인가*: 계산 수행 또는 외부 데이터 액세스와 같이 LLM에 추가 기능을 제공하는 함수.
-   *도구를 정의하는 방법*: 명확한 텍스트 설명, 입력, 출력 및 호출 가능한 함수를 제공함으로써.
-   *도구가 필수적인 이유*: 에이전트가 정적 모델 훈련의 한계를 극복하고, 실시간 작업을 처리하며, 전문화된 행동을 수행할 수 있도록 합니다.

이제 [에이전트 워크플로우](agent-steps-and-structure)로 넘어갈 수 있습니다. 여기서는 에이전트가 어떻게 관찰하고, 생각하고, 행동하는지 볼 수 있습니다. 이는 **지금까지 다룬 모든 것을 통합**하고 완전히 기능하는 AI 에이전트를 만드는 단계를 설정합니다.

하지만 먼저, 짧은 퀴즈 시간입니다!