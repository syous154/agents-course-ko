# 빠른 자체 점검 (미채점) [[quiz2]]

또 퀴즈라고요? 알아요, 알아요, ... 😅 하지만 이 짧은 미채점 퀴즈는 **방금 학습한 핵심 개념을 강화하는 데 도움을 주기 위해** 여기에 있습니다.

이 퀴즈는 효과적인 AI 에이전트를 구축하는 데 필수적인 구성 요소인 에이전트 워크플로우 및 상호 작용을 다룹니다.

### Q1: LlamaIndex에서 AgentWorkflow의 목적은 무엇인가요?

<Question
choices={[
{
text: "하나 이상의 에이전트를 도구와 함께 실행하기 위함",
explain: "네, AgentWorkflow는 하나 이상의 에이전트로 시스템을 빠르게 생성하는 주요 방법입니다.",
correct: true
},
{
text: "메모리 없이 데이터를 쿼리할 수 있는 단일 에이전트를 생성하기 위함",
explain: "아니요, AgentWorkflow는 그보다 더 많은 기능을 가지고 있으며, QueryEngine은 데이터에 대한 간단한 쿼리를 위한 것입니다.",
},
{
text: "에이전트를 위한 도구를 자동으로 구축하기 위함",
explain: "AgentWorkflow는 도구를 구축하지 않습니다. 그것은 개발자의 역할입니다.",
},
{
text: "에이전트 메모리 및 상태를 관리하기 위함",
explain: "메모리 및 상태 관리는 AgentWorkflow의 주요 목적이 아닙니다.",
}
]}
/>

---

### Q2: 워크플로우의 상태를 추적하는 데 사용되는 객체는 무엇인가요?

<Question
choices={[
{
text: "State",
explain: "State는 워크플로우 상태 관리를 위한 올바른 객체가 아닙니다.",
},
{
text: "Context",
explain: "Context는 워크플로우 상태를 추적하는 데 사용되는 올바른 객체입니다.",
correct: true
},
{
text: "WorkflowState",
explain: "WorkflowState는 올바른 객체가 아닙니다.",
},
{
text: "Management",
explain: "Management는 워크플로우 상태에 유효한 객체가 아닙니다.",
}
]}
/>

---

### Q3: 에이전트가 이전 상호 작용을 기억하게 하려면 어떤 메서드를 사용해야 하나요?

<Question
choices={[
{
text: "run(query_str)",
explain: ".run(query_str)은 대화 기록을 유지하지 않습니다.",
},
{
text: "chat(query_str, ctx=ctx)",
explain: "chat()은 워크플로우에서 유효한 메서드가 아닙니다.",
},
{
text: "interact(query_str)",
explain: "interact()은 에이전트 상호 작용에 유효한 메서드가 아닙니다.",
},
{
text: "run(query_str, ctx=ctx)",
explain: "컨텍스트를 전달하고 유지함으로써 상태를 유지할 수 있습니다!",
correct: true
}
]}
/>

---

### Q4: Agentic RAG의 주요 특징은 무엇인가요?

<Question
choices={[
{
text: "RAG 워크플로우에서 질문에 답하기 위해 문서 기반 도구만 사용할 수 있습니다.",
explain: "Agentic RAG는 문서 기반 도구를 포함하여 다양한 도구를 사용할 수 있습니다.",
},
{
text: "챗봇처럼 도구 없이 자동으로 질문에 답합니다.",
explain: "Agentic RAG는 질문에 답하기 위해 도구를 사용합니다.",
},
{
text: "RAG 도구를 포함하여 질문에 답하기 위해 어떤 도구든 사용할 수 있습니다.",
explain: "Agentic RAG는 질문에 답하기 위해 다양한 도구를 유연하게 사용할 수 있습니다.",
correct: true
},
{
text: "함수 호출 에이전트에서만 작동합니다.",
explain: "Agentic RAG는 함수 호출 에이전트에만 국한되지 않습니다.",
}
]}
/>

---

이해하셨나요? 좋습니다! 이제 **단원의 간략한 요약을 해봅시다!**