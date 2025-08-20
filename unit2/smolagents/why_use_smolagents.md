smolagents 배너 이미지

# smolagents를 사용하는 이유

이 모듈에서는 [smolagents](https://huggingface.co/docs/smolagents/en/index) 사용의 장단점을 살펴보고, smolagents가 귀하의 필요에 적합한 프레임워크인지에 대한 정보에 입각한 결정을 내리는 데 도움을 드릴 것입니다.

## `smolagents`란 무엇인가요?

`smolagents`는 AI 에이전트 구축을 위한 간단하면서도 강력한 프레임워크입니다. 이는 LLM(대규모 언어 모델)에 검색 또는 이미지 생성과 같은 실제 세계와 상호 작용할 수 있는 _에이전시_를 제공합니다.

1단원에서 배웠듯이, AI 에이전트는 **'관찰'**을 기반으로 **'생각'**을 생성하여 **'행동'**을 수행하기 위해 LLM을 사용하는 프로그램입니다. smolagents에서 이것이 어떻게 구현되는지 살펴보겠습니다.

### `smolagents`의 주요 장점
- **단순성:** 최소한의 코드 복잡성과 추상화로 프레임워크를 이해하고 채택하며 확장하기 쉽게 만듭니다.
- **유연한 LLM 지원:** Hugging Face 도구 및 외부 API와의 통합을 통해 모든 LLM과 작동합니다.
- **코드 우선 접근 방식:** 작업을 코드로 직접 작성하는 코드 에이전트에 대한 일등 지원을 통해 파싱 필요성을 없애고 도구 호출을 단순화합니다.
- **HF Hub 통합:** Hugging Face Hub와의 원활한 통합을 통해 Gradio Spaces를 도구로 사용할 수 있습니다.

### smolagents는 언제 사용해야 할까요?

이러한 장점을 고려할 때, 다른 프레임워크 대신 smolagents를 언제 사용해야 할까요?

smolagents는 다음과 같은 경우에 이상적입니다.
- **경량의 최소한의 솔루션**이 필요할 때.
- 복잡한 구성 없이 **빠르게 실험**하고 싶을 때.
- **애플리케이션 로직이 간단**할 때.

### 코드 vs. JSON 액션
에이전트가 JSON으로 액션을 작성하는 다른 프레임워크와 달리, `smolagents`는 **코드 내 도구 호출에 중점**을 두어 실행 프로세스를 단순화합니다. 이는 도구를 호출하는 코드를 빌드하기 위해 JSON을 파싱할 필요가 없기 때문입니다. 출력은 직접 실행될 수 있습니다.

다음 다이어그램은 이러한 차이점을 보여줍니다.

코드 vs. JSON 액션 이미지

코드 vs. JSON 액션의 차이점을 다시 살펴보려면 [1단원의 액션 섹션](https://huggingface.co/learn/agents-course/unit1/actions#actions-enabling-the-agent-to-engage-with-its-environment)을 다시 방문할 수 있습니다.

### `smolagents`의 에이전트 유형

`smolagents`의 에이전트는 **다단계 에이전트**로 작동합니다.

각 [`MultiStepAgent`](https://huggingface.co/docs/smolagents/main/en/reference/agents#smolagents.MultiStepAgent)는 다음을 수행합니다.
- 하나의 생각
- 하나의 도구 호출 및 실행

주요 에이전트 유형으로 **[CodeAgent](https://huggingface.co/docs/smolagents/main/en/reference/agents#smolagents.CodeAgent)**를 사용하는 것 외에도, smolagents는 JSON으로 도구 호출을 작성하는 **[ToolCallingAgent](https://huggingface.co/docs/smolagents/main/en/reference/agents#smolagents.ToolCallingAgent)**도 지원합니다.

다음 섹션에서 각 에이전트 유형을 더 자세히 살펴보겠습니다.

<Tip>
smolagents에서 도구는 Python 함수를 래핑하는 <code>@tool</code> 데코레이터 또는 <code>Tool</code> 클래스를 사용하여 정의됩니다.
</Tip>

### `smolagents`의 모델 통합
`smolagents`는 유연한 LLM 통합을 지원하여 [특정 기준](https://huggingface.co/docs/smolagents/main/en/reference/models)을 충족하는 모든 호출 가능한 모델을 사용할 수 있도록 합니다. 이 프레임워크는 모델 연결을 단순화하기 위해 여러 사전 정의된 클래스를 제공합니다.

- **[TransformersModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.TransformersModel):** 원활한 통합을 위해 로컬 `transformers` 파이프라인을 구현합니다.
- **[InferenceClientModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.InferenceClientModel):** [Hugging Face의 인프라](https://huggingface.co/docs/api-inference/index) 또는 증가하는 [타사 추론 제공업체](https://huggingface.co/docs/huggingface_hub/main/en/guides/inference#supported-providers-and-tasks)를 통해 [서버리스 추론](https://huggingface.co/docs/huggingface_hub/main/en/guides/inference) 호출을 지원합니다.
- **[LiteLLMModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.LiteLLMModel):** 경량 모델 상호 작용을 위해 [LiteLLM](https://www.litellm.ai/)을 활용합니다.
- **[OpenAIServerModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.OpenAIServerModel):** OpenAI API 인터페이스를 제공하는 모든 서비스에 연결합니다.
- **[AzureOpenAIServerModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.AzureOpenAIServerModel):** 모든 Azure OpenAI 배포와의 통합을 지원합니다.

이러한 유연성은 개발자가 특정 사용 사례에 가장 적합한 모델과 서비스를 선택할 수 있도록 보장하며, 쉬운 실험을 가능하게 합니다.

이제 smolagents를 사용하는 이유와 시기를 이해했으니, 이 강력한 라이브러리에 대해 더 자세히 알아보겠습니다!

## 자료

- [smolagents 블로그](https://huggingface.co/blog/smolagents) - smolagents 및 코드 상호 작용 소개