# `smolagents` 소개

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/smolagents/thumbnail.jpg" alt="Unit 2.1 Thumbnail"/>

이 모듈에 오신 것을 환영합니다. 여기서는 유능한 AI 에이전트 생성을 위한 경량 프레임워크를 제공하는 [`smolagents`](https://github.com/huggingface/smolagents) 라이브러리를 사용하여 **효과적인 에이전트를 구축하는 방법**을 배우게 됩니다.

`smolagents`는 Hugging Face 라이브러리입니다. 따라서 smolagents [`저장소`](https://github.com/huggingface/smolagents)에 **별표**를 눌러 지원해 주시면 감사하겠습니다.
<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/smolagents/star_smolagents.gif" alt="staring smolagents"/>

## 모듈 개요

이 모듈은 `smolagents`를 사용하여 지능형 에이전트를 구축하기 위한 주요 개념과 실용적인 전략에 대한 포괄적인 개요를 제공합니다.

다양한 오픈 소스 프레임워크가 있는 상황에서, `smolagents`를 유용한 옵션으로 만드는 구성 요소와 기능을 이해하거나 다른 솔루션이 더 적합한 시기를 결정하는 것이 중요합니다.

소프트웨어 개발 작업을 위해 설계된 코드 에이전트, 모듈식 함수 기반 워크플로우 생성을 위한 도구 호출 에이전트, 정보에 접근하고 통합하는 검색 에이전트를 포함한 중요한 에이전트 유형을 탐색할 것입니다.

또한, 여러 에이전트의 오케스트레이션, 뿐만 아니라 동적이고 상황 인식적인 애플리케이션을 위한 새로운 가능성을 열어주는 비전 기능 및 웹 브라우징 통합에 대해서도 다룰 것입니다.

이번 유닛에서는 유닛 1의 에이전트인 알프레드가 다시 등장합니다. 이번에는 `smolagents` 프레임워크를 내부 작동에 사용합니다. 알프레드가 다양한 작업을 처리하는 동안 이 프레임워크의 주요 개념을 함께 탐색할 것입니다. 알프레드는 웨인 가족 🦇이 없는 동안 웨인 저택에서 파티를 준비하고 있으며, 할 일이 많습니다. 알프레드의 여정과 그가 `smolagents`로 이러한 작업을 어떻게 처리하는지 함께 살펴보세요!

<Tip>

이 유닛에서는 `smolagents` 라이브러리로 AI 에이전트를 구축하는 방법을 배우게 됩니다. 여러분의 에이전트는 데이터를 검색하고, 코드를 실행하고, 웹 페이지와 상호 작용할 수 있습니다. 또한 여러 에이전트를 결합하여 더 강력한 시스템을 만드는 방법도 배우게 됩니다.

</Tip>

![Alfred the agent](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/this-is-alfred.jpg)

## 목차

`smolagents`에 대한 이 유닛에서는 다음 내용을 다룹니다.

### 1️⃣ [smolagents를 사용하는 이유](./why_use_smolagents)

`smolagents`는 애플리케이션 개발에 사용할 수 있는 많은 오픈 소스 에이전트 프레임워크 중 하나입니다. 대체 옵션으로는 이 과정의 다른 모듈에서도 다루는 `LlamaIndex` 및 `LangGraph`가 있습니다. `smolagents`는 특정 사용 사례에 매우 적합할 수 있는 몇 가지 주요 기능을 제공하지만, 프레임워크를 선택할 때는 항상 모든 옵션을 고려해야 합니다. `smolagents` 사용의 장단점을 탐색하여 프로젝트 요구 사항에 따라 정보에 입각한 결정을 내릴 수 있도록 돕습니다.

### 2️⃣ [CodeAgents](./code_agents)

`CodeAgents`는 `smolagents`의 주요 에이전트 유형입니다. 이 에이전트는 JSON이나 텍스트를 생성하는 대신 Python 코드를 생성하여 작업을 수행합니다. 이 모듈에서는 CodeAgents의 목적, 기능 및 작동 방식과 함께 기능을 보여주는 실습 예제를 탐색합니다.

### 3️⃣ [ToolCallingAgents](./tool_calling_agents)

`ToolCallingAgents`는 `smolagents`에서 지원하는 두 번째 에이전트 유형입니다. Python 코드를 생성하는 `CodeAgents`와 달리, 이 에이전트는 시스템이 작업을 실행하기 위해 구문 분석하고 해석해야 하는 JSON/텍스트 블롭에 의존합니다. 이 모듈에서는 ToolCallingAgents의 기능, `CodeAgents`와의 주요 차이점, 그리고 사용법을 설명하는 예제를 제공합니다.

### 4️⃣ [Tools](./tools)

유닛 1에서 보았듯이, 도구는 LLM이 에이전트 시스템 내에서 사용할 수 있는 함수이며, 에이전트 동작의 필수적인 구성 요소 역할을 합니다. 이 모듈에서는 `Tool` 클래스 또는 `@tool` 데코레이터를 사용하여 도구를 생성하는 방법, 도구의 구조 및 다양한 구현 방법을 다룹니다. 또한 기본 도구 상자, 커뮤니티와 도구를 공유하는 방법, 그리고 에이전트에서 사용할 커뮤니티 기여 도구를 로드하는 방법에 대해서도 배웁니다.

### 5️⃣ [Retrieval Agents](./retrieval_agents)

검색 에이전트는 모델이 지식 베이스에 접근할 수 있도록 하여 여러 소스에서 정보를 검색, 통합 및 추출할 수 있도록 합니다. 이들은 효율적인 검색을 위해 벡터 저장소를 활용하고 **검색 증강 생성(RAG)** 패턴을 구현합니다. 이러한 에이전트는 메모리 시스템을 통해 대화 컨텍스트를 유지하면서 웹 검색을 사용자 지정 지식 베이스와 통합하는 데 특히 유용합니다. 이 모듈에서는 강력한 정보 검색을 위한 대체 메커니즘을 포함한 구현 전략을 탐색합니다.

### 6️⃣ [Multi-Agent Systems](./multi_agent_systems)

여러 에이전트를 효과적으로 오케스트레이션하는 것은 강력한 다중 에이전트 시스템을 구축하는 데 중요합니다. 웹 검색 에이전트와 코드 실행 에이전트와 같이 서로 다른 기능을 가진 에이전트를 결합함으로써 더 정교한 솔루션을 만들 수 있습니다. 이 모듈은 효율성과 신뢰성을 극대화하기 위해 다중 에이전트 시스템을 설계, 구현 및 관리하는 데 중점을 둡니다.

### 7️⃣ [Vision and Browser agents](./vision_agents)

비전 에이전트는 **시각-언어 모델(VLM)**을 통합하여 시각 정보를 처리하고 해석할 수 있도록 함으로써 기존 에이전트 기능을 확장합니다. 이 모듈은 이미지 기반 추론, 시각 데이터 분석 및 다중 모달 상호 작용과 같은 고급 기능을 잠금 해제하는 VLM 기반 에이전트를 설계하고 통합하는 방법을 탐색합니다. 또한 비전 에이전트를 사용하여 웹을 탐색하고 정보를 추출할 수 있는 브라우저 에이전트를 구축할 것입니다.

## 자료

- [smolagents 문서](https://huggingface.co/docs/smolagents) - smolagents 라이브러리 공식 문서
- [효과적인 에이전트 구축](https://www.anthropic.com/research/building-effective-agents) - 에이전트 아키텍처에 대한 연구 논문
- [에이전트 가이드라인](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - 신뢰할 수 있는 에이전트 구축을 위한 모범 사례
- [LangGraph 에이전트](https://langchain-ai.github.io/langgraph/) - 에이전트 구현의 추가 예제
- [함수 호출 가이드](https://platform.openai.com/docs/guides/function-calling) - LLM의 함수 호출 이해
- [RAG 모범 사례](https://www.pinecone.io/learn/retrieval-augmented-generation/) - 효과적인 RAG 구현 가이드