# 사고-행동-관찰 주기를 통해 AI 에이전트 이해하기

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/whiteboard-check-3.jpg" alt="Unit 1 planning"/>

이전 섹션에서는 다음을 배웠습니다.

- **시스템 프롬프트에서 에이전트에게 도구가 어떻게 제공되는지.**
- **AI 에이전트가 '추론'하고, 계획하며, 환경과 상호작용할 수 있는 시스템이라는 점.**

이 섹션에서는 **완전한 AI 에이전트 워크플로우, 즉 사고-행동-관찰(Thought-Action-Observation)로 정의한 주기를 탐구할 것입니다.**

그리고 각 단계에 대해 더 깊이 파고들 것입니다.

## 핵심 구성 요소

에이전트의 작업은 **생각하기(사고) → 행동하기(행동) → 관찰하기(관찰)**의 연속적인 주기입니다.

이러한 행동들을 함께 살펴보겠습니다.

1.  **사고(Thought)**: 에이전트의 LLM(대규모 언어 모델) 부분이 다음 단계가 무엇이 되어야 할지 결정합니다.
2.  **행동(Action)**: 에이전트는 관련 인수를 사용하여 도구를 호출함으로써 행동을 취합니다.
3.  **관찰(Observation)**: 모델은 도구의 응답을 반영합니다.

## 사고-행동-관찰 주기

세 가지 구성 요소는 연속적인 루프에서 함께 작동합니다. 프로그래밍의 비유를 사용하자면, 에이전트는 **while 루프**를 사용합니다. 즉, 에이전트의 목표가 달성될 때까지 루프가 계속됩니다.

시각적으로는 다음과 같습니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/AgentCycle.gif" alt="Think, Act, Observe cycle"/>

많은 에이전트 프레임워크에서 **규칙과 지침은 시스템 프롬프트에 직접 내장되어** 모든 주기가 정의된 논리를 따르도록 보장합니다.

간단한 버전에서는 시스템 프롬프트가 다음과 같을 수 있습니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/system_prompt_cycle.png" alt="Think, Act, Observe cycle"/>

여기서 시스템 메시지에서 다음을 정의했음을 알 수 있습니다.

-   *에이전트의 행동*.
-   이전 섹션에서 설명했듯이 *에이전트가 접근할 수 있는 도구*.
-   LLM 지침에 포함된 *사고-행동-관찰 주기*.

각 프로세스 단계에 더 깊이 들어가기 전에 프로세스를 이해하기 위한 작은 예시를 살펴보겠습니다.

## 날씨 에이전트 알프레드

우리는 날씨 에이전트 알프레드를 만들었습니다.

사용자가 알프레드에게 묻습니다: "뉴욕의 현재 날씨는 어떤가요?"

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/alfred-agent.jpg" alt="Alfred Agent"/>

알프레드의 임무는 날씨 API 도구를 사용하여 이 질문에 답하는 것입니다.

주기가 어떻게 전개되는지 살펴보겠습니다.

### 사고(Thought)

**내부 추론:**

질문을 받은 알프레드의 내부 대화는 다음과 같을 수 있습니다.

*"사용자는 뉴욕의 현재 날씨 정보가 필요합니다. 저는 날씨 데이터를 가져오는 도구에 접근할 수 있습니다. 먼저, 최신 정보를 얻기 위해 날씨 API를 호출해야 합니다."*

이 단계는 에이전트가 문제를 단계별로 나누는 것을 보여줍니다. 첫째, 필요한 데이터를 수집합니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/alfred-agent-1.jpg" alt="Alfred Agent"/>

### 행동(Action)

**도구 사용:**

알프레드는 자신의 추론과 `get_weather` 도구에 대해 알고 있다는 사실을 바탕으로 날씨 API 도구를 호출하는 JSON 형식의 명령을 준비합니다. 예를 들어, 첫 번째 행동은 다음과 같을 수 있습니다.

사고: 뉴욕의 현재 날씨를 확인해야 합니다.

```
    {
      "action": "get_weather",
      "action_input": {
        "location": "New York"
      }
    }
```

여기서 행동은 호출할 도구(예: get_weather)와 전달할 매개변수("location": "New York")를 명확하게 지정합니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/alfred-agent-2.jpg" alt="Alfred Agent"/>

### 관찰(Observation)

**환경으로부터의 피드백:**

도구 호출 후 알프레드는 관찰 결과를 받습니다. 이는 다음과 같은 API의 원시 날씨 데이터일 수 있습니다.

*"뉴욕의 현재 날씨: 부분적으로 흐림, 15°C, 습도 60%."*

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/alfred-agent-3.jpg" alt="Alfred Agent"/>

이 관찰 결과는 추가 컨텍스트로 프롬프트에 추가됩니다. 이는 행동이 성공했는지 확인하고 필요한 세부 정보를 제공하는 실제 피드백 역할을 합니다.

### 업데이트된 사고(thought)

**반영:**

관찰 결과를 손에 쥔 알프레드는 내부 추론을 업데이트합니다.

*"이제 뉴욕의 날씨 데이터를 얻었으니, 사용자에게 답변을 작성할 수 있습니다."*

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/alfred-agent-4.jpg" alt="Alfred Agent"/>

### 최종 행동(Action)

알프레드는 우리가 지시한 형식으로 최종 응답을 생성합니다.

사고: 이제 날씨 데이터가 있습니다. 뉴욕의 현재 날씨는 부분적으로 흐리고 기온은 15°C이며 습도는 60%입니다."

최종 답변: 뉴욕의 현재 날씨는 부분적으로 흐리고 기온은 15°C이며 습도는 60%입니다.

이 최종 행동은 사용자에게 답변을 다시 보내 루프를 닫습니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/alfred-agent-5.jpg" alt="Alfred Agent"/>

이 예시에서 우리가 본 것:

-   **에이전트는 목표가 달성될 때까지 루프를 반복합니다.**

    **알프레드의 프로세스는 순환적입니다.** 생각으로 시작하여 도구를 호출하여 행동하고, 마지막으로 결과를 관찰합니다. 만약 관찰 결과가 오류나 불완전한 데이터를 나타냈다면, 알프레드는 접근 방식을 수정하기 위해 주기를 다시 시작할 수 있었습니다.

-   **도구 통합:**

    도구(예: 날씨 API)를 호출하는 능력은 알프레드가 **정적 지식을 넘어 실시간 데이터를 검색**할 수 있도록 하며, 이는 많은 AI 에이전트의 필수적인 측면입니다.

-   **동적 적응:**

    각 주기는 에이전트가 새로운 정보(관찰)를 추론(사고)에 통합하여 최종 답변이 충분한 정보를 바탕으로 정확하도록 보장합니다.

이 예시는 *ReAct 주기*(다음 섹션에서 다룰 개념)의 핵심 개념을 보여줍니다. 즉, **사고, 행동, 관찰의 상호작용은 AI 에이전트가 복잡한 작업을 반복적으로 해결할 수 있도록 합니다.**

이러한 원칙을 이해하고 적용함으로써, 여러분은 자신의 작업을 추론할 뿐만 아니라 **외부 도구를 효과적으로 활용하여 작업을 완료**하고, 환경 피드백을 기반으로 출력을 지속적으로 개선하는 에이전트를 설계할 수 있습니다.

---

이제 프로세스의 개별 단계로서 사고, 행동, 관찰에 대해 더 깊이 파고들어 보겠습니다.