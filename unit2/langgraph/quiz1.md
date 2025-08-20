# LangGraph 이해도 테스트

LangGraph에 대한 이해도를 간단한 퀴즈로 테스트해 봅시다! 이는 지금까지 다룬 핵심 개념을 강화하는 데 도움이 될 것입니다.

이 퀴즈는 선택 사항이며 채점되지 않습니다.

### Q1: LangGraph의 주요 목적은 무엇입니까?
LangGraph가 무엇을 위해 설계되었는지 가장 잘 설명하는 문장은 무엇입니까?

<Question
choices={[
  {
    text: "LLM을 포함하는 애플리케이션의 제어 흐름을 구축하기 위한 프레임워크",
    explain: "LangGraph는 LLM을 사용하는 애플리케이션의 제어 흐름을 구축하고 관리하는 데 특별히 설계되었습니다.",
    correct: true
  },
  {
    text: "다양한 LLM 모델과 상호 작용하기 위한 인터페이스를 제공하는 라이브러리",
    explain: "이는 모델 상호 작용을 위한 표준 인터페이스를 제공하는 LangChain의 역할을 더 잘 설명합니다. LangGraph는 제어 흐름에 중점을 둡니다.",
  },
  {
    text: "도구 호출을 위한 에이전트 라이브러리",
    explain: "While LangGraph works with agents, the main purpose of langGraph is 'Ochestration'.",
  }
]}
/>

---

### Q2: "제어 대 자유"의 균형이라는 맥락에서 LangGraph는 어떤 위치에 있습니까?
LangGraph의 에이전트 설계 접근 방식을 가장 잘 특징짓는 문장은 무엇입니까?

<Question
choices={[
  {
    text: "LangGraph는 LLM이 모든 결정을 독립적으로 내릴 수 있도록 자유를 극대화합니다.",
    explain: "LangGraph는 실제로 자유보다는 제어에 더 중점을 두어 LLM 워크플로에 대한 구조를 제공합니다.",
  },
  {
    text: "LangGraph는 의사 결정을 위해 LLM 기능을 활용하면서도 실행 흐름에 대한 강력한 제어를 제공합니다.",
    explain: "LangGraph는 에이전트의 실행을 제어해야 할 때 빛을 발하며, 구조화된 워크플로를 통해 예측 가능한 동작을 제공합니다.",
    correct: true
  },
]}
/>

---

### Q3: LangGraph에서 상태(State)는 어떤 역할을 합니까?
LangGraph에서 상태(State)에 대한 가장 정확한 설명을 선택하십시오.

<Question
choices={[
  {
    text: "상태는 LLM의 최신 생성물입니다.",
    explain: "상태는 LLM이 생성한 것이 아니라 LangGraph의 사용자 정의 클래스입니다. 필드는 사용자 정의이며, 값은 LLM으로 채워질 수 있습니다.",
  },
  {
    text: "상태는 실행 중 오류를 추적하는 데만 사용됩니다.",
    explain: "상태는 오류 추적보다 훨씬 더 광범위한 목적을 가지고 있습니다. 하지만 여전히 유용합니다.",
  },
  {
    text: "상태는 에이전트 애플리케이션을 통해 흐르는 정보를 나타냅니다.",
    explain: "상태는 LangGraph의 핵심이며 단계 간 의사 결정을 위해 필요한 모든 정보를 포함합니다. 계산해야 하는 필드를 제공하고 노드는 분기를 결정하기 위해 값을 변경할 수 있습니다.",
    correct: true
  },
  {
    text: "상태는 외부 API와 작업할 때만 관련이 있습니다.",
    explain: "상태는 외부 API와 작업하는 애플리케이션뿐만 아니라 모든 LangGraph 애플리케이션의 기본입니다.",
  }
]}
/>

### Q4: LangGraph에서 조건부 엣지(Conditional Edge)는 무엇입니까?
가장 정확한 설명을 선택하십시오.

<Question
choices={[
    {
    text: "조건 평가를 기반으로 다음에 실행할 노드를 결정하는 엣지",
    explain: "조건부 엣지를 사용하면 그래프가 현재 상태를 기반으로 동적 라우팅 결정을 내리고 워크플로에서 분기 논리를 생성할 수 있습니다.",
    correct: true
  },
  {
    text: "특정 조건이 발생할 때만 따라가는 엣지",
    explain: "조건부 엣지는 입력이 아닌 출력에 대한 애플리케이션의 흐름을 제어합니다.",
  },
  {
    text: "진행하기 전에 사용자 확인이 필요한 엣지",
    explain: "조건부 엣지는 프로그래밍 방식의 조건에 기반하며 사용자 상호 작용 요구 사항에 기반하지 않습니다.",
  }
]}
/>

---

### Q5: LangGraph는 LLM의 환각 문제를 해결하는 데 어떻게 도움이 됩니까?
가장 적절한 답변을 선택하십시오.

<Question
choices={[
  {
    text: "LangGraph는 LLM 응답을 제한하여 환각을 완전히 제거합니다.",
    explain: "어떤 프레임워크도 LLM의 환각을 완전히 제거할 수는 없으며, LangGraph도 예외는 아닙니다.",
  },
  {
    text: "LangGraph는 LLM 출력을 검증하고 확인하는 구조화된 워크플로를 제공합니다.",
    explain: "유효성 검사 단계, 확인 노드 및 오류 처리 경로가 있는 구조화된 워크플로를 생성함으로써 LangGraph는 환각의 영향을 줄이는 데 도움이 됩니다.",
    correct: true
  },
  {
    text: "LangGraph는 환각에 영향을 미치지 않습니다.",
    explain: "LangGraph의 구조화된 워크플로 접근 방식은 속도 저하를 감수하고서라도 환각을 완화하는 데 크게 도움이 될 수 있습니다.",
  }
]}
/>

퀴즈를 완료하신 것을 축하드립니다! 🎉 질문을 놓친 부분이 있다면 이전 섹션을 검토하여 이해도를 높이십시오. 다음으로, LangGraph의 고급 기능을 탐색하고 더 복잡한 에이전트 워크플로를 구축하는 방법을 살펴보겠습니다.