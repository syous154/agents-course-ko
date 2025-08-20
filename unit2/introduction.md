# 에이전트 프레임워크 소개

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/thumbnail.jpg" alt="썸네일"/>

강력한 에이전트 애플리케이션을 구축하는 데 사용할 수 있는 **다양한 에이전트 프레임워크를 탐색할** 두 번째 유닛에 오신 것을 환영합니다.

우리는 다음을 학습할 것입니다:

- 유닛 2.1: [smolagents](https://huggingface.co/docs/smolagents/en/index)
- 유닛 2.2: [LlamaIndex](https://www.llamaindex.ai/)
- 유닛 2.3: [LangGraph](https://www.langchain.com/langgraph)

시작해 봅시다! 🕵

## 에이전트 프레임워크를 사용해야 하는 경우

LLM을 중심으로 애플리케이션을 구축할 때 에이전트 프레임워크가 **항상 필요한 것은 아닙니다**. 에이전트 프레임워크는 특정 작업을 효율적으로 해결하기 위한 워크플로우의 유연성을 제공하지만, 항상 필수적인 것은 아닙니다.

때로는 **사전 정의된 워크플로우만으로도** 사용자 요청을 충족하기에 충분하며, 에이전트 프레임워크가 실제로 필요하지 않을 수도 있습니다. 프롬프트 체인처럼 에이전트를 구축하는 접근 방식이 간단하다면, 일반 코드를 사용하는 것만으로도 충분할 수 있습니다. 이점은 개발자가 **추상화 없이 시스템을 완전히 제어하고 이해할 수 있다는** 것입니다.

그러나 LLM이 함수를 호출하거나 여러 에이전트를 사용하는 것과 같이 워크플로우가 더 복잡해지면 이러한 추상화가 도움이 되기 시작합니다.

이러한 아이디어를 고려할 때, 우리는 이미 몇 가지 기능의 필요성을 확인할 수 있습니다:

* 시스템을 구동하는 *LLM 엔진*.
* 에이전트가 접근할 수 있는 *도구 목록*.
* LLM 출력에서 도구 호출을 추출하기 위한 *파서*.
* 파서와 동기화된 *시스템 프롬프트*.
* *메모리 시스템*.
* LLM 실수를 제어하기 위한 *오류 로깅 및 재시도 메커니즘*.

`smolagents`, `LlamaIndex`, `LangGraph`를 포함한 다양한 프레임워크에서 이러한 주제가 어떻게 해결되는지 탐색할 것입니다.

## 에이전트 프레임워크 유닛

| 프레임워크 | 설명 | 유닛 저자 |
|---|---|---|
| [smolagents](./smolagents/introduction) | Hugging Face에서 개발한 에이전트 프레임워크. | Sergio Paniego - [HF](https://huggingface.co/sergiopaniego) - [X](https://x.com/sergiopaniego) - [Linkedin](https://www.linkedin.com/in/sergio-paniego-blanco) |
| [Llama-Index](./llama-index/introduction) | 컨텍스트 증강 AI 에이전트를 프로덕션에 배포하기 위한 엔드투엔드 툴링 | David Berenstein - [HF](https://huggingface.co/davidberenstein1957) - [X](https://x.com/davidberenstei) - [Linkedin](https://www.linkedin.com/in/davidberenstein) |
| [LangGraph](./langgraph/introduction) | 에이전트의 상태 저장 오케스트레이션을 허용하는 에이전트 | Joffrey THOMAS - [HF](https://huggingface.co/Jofthomas) - [X](https://x.com/Jthmas404) - [Linkedin](https://www.linkedin.com/in/joffrey-thomas) |