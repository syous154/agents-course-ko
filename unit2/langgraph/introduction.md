# `LangGraph` 소개

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/LangGraph.png" alt="Unit 2.3 썸네일"/>

복잡한 LLM 워크플로우를 구조화하고 오케스트레이션하는 데 도움이 되도록 설계된 [`LangGraph`](https://github.com/langchain-ai/langgraph) 프레임워크를 사용하여 **애플리케이션을 구축하는 방법**을 배우게 될 여정의 다음 부분에 오신 것을 환영합니다.

`LangGraph`는 에이전트의 흐름을 **제어**하는 도구를 제공하여 **프로덕션 준비가 된** 애플리케이션을 구축할 수 있게 해주는 프레임워크입니다.

## 모듈 개요

이 유닛에서는 다음을 다룹니다:

### 1️⃣ [LangGraph란 무엇이며, 언제 사용해야 할까요?](./when_to_use_langgraph)
### 2️⃣ [LangGraph의 구성 요소](./building_blocks)
### 3️⃣ [알프레드, 메일 분류 집사](./first_graph)
### 4️⃣ [알프레드, 문서 분석 에이전트](./document_analysis_agent)
### 5️⃣ [퀴즈](./quizz1)

<Tip warning={true}>
이 섹션의 예제는 강력한 LLM/VLM 모델에 대한 액세스를 필요로 합니다. LangGraph와 가장 호환성이 좋기 때문에 GPT-4o API를 사용하여 실행했습니다.
</Tip>

이 유닛을 마치면 견고하고 체계적이며 프로덕션 준비가 된 애플리케이션을 구축할 수 있게 될 것입니다!

이 섹션은 LangGraph에 대한 소개이며, 더 고급 주제는 무료 LangChain 아카데미 코스인 [LangGraph 소개](https://academy.langchain.com/courses/intro-to-langgraph)에서 확인할 수 있습니다.

시작해 봅시다!

## 자료

- [LangGraph 에이전트](https://langchain-ai.github.io/langgraph/) - LangGraph 에이전트 예제
- [LangChain 아카데미](https://academy.langchain.com/courses/intro-to-langgraph) - LangChain의 LangGraph 전체 코스