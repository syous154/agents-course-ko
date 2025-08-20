# LLM이란 무엇인가요?

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/whiteboard-check-1.jpg" alt="Unit 1 planning"/>

이전 섹션에서는 각 에이전트의 핵심에 AI 모델이 필요하며, LLM이 이러한 목적에 가장 일반적인 유형의 AI 모델이라는 것을 배웠습니다.

이제 LLM이 무엇이며 에이전트에 어떻게 동력을 공급하는지 알아보겠습니다.

이 섹션에서는 LLM 사용에 대한 간결한 기술적 설명을 제공합니다. 더 깊이 탐구하고 싶다면, 저희의 <a href="https://huggingface.co/learn/nlp-course/chapter1/1" target="_blank">무료 자연어 처리 강좌</a>를 확인해 보세요.

## 대규모 언어 모델(LLM)이란 무엇인가요?

LLM은 인간의 언어를 이해하고 생성하는 데 탁월한 AI 모델 유형입니다. 방대한 양의 텍스트 데이터로 훈련되어 언어의 패턴, 구조, 심지어 뉘앙스까지 학습할 수 있습니다. 이러한 모델은 일반적으로 수백만 개의 매개변수로 구성됩니다.

오늘날 대부분의 LLM은 트랜스포머 아키텍처를 기반으로 구축됩니다. 이는 2018년 Google에서 BERT가 출시된 이후 상당한 관심을 얻은 "어텐션(Attention)" 알고리즘을 기반으로 하는 딥러닝 아키텍처입니다.

<figure>
<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/transformer.jpg" alt="Transformer"/>
<figcaption>원래 트랜스포머 아키텍처는 왼쪽의 인코더와 오른쪽의 디코더로 구성되어 있었습니다.
</figcaption>
</figure>

트랜스포머에는 3가지 유형이 있습니다.

1.  **인코더**
    인코더 기반 트랜스포머는 텍스트(또는 기타 데이터)를 입력으로 받아 해당 텍스트의 밀집된 표현(또는 임베딩)을 출력합니다.
    *   **예시**: Google의 BERT
    *   **사용 사례**: 텍스트 분류, 의미 검색, 개체명 인식
    *   **일반적인 크기**: 수백만 개의 매개변수

2.  **디코더**
    디코더 기반 트랜스포머는 한 번에 하나의 토큰씩 시퀀스를 완성하기 위해 새로운 토큰을 생성하는 데 중점을 둡니다.
    *   **예시**: Meta의 Llama
    *   **사용 사례**: 텍스트 생성, 챗봇, 코드 생성
    *   **일반적인 크기**: 수십억(미국 기준, 즉 10^9) 개의 매개변수

3.  **Seq2Seq (인코더-디코더)**
    시퀀스-투-시퀀스 트랜스포머는 인코더와 디코더를 결합합니다. 인코더가 먼저 입력 시퀀스를 컨텍스트 표현으로 처리한 다음, 디코더가 출력 시퀀스를 생성합니다.
    *   **예시**: T5, BART
    *   **사용 사례**: 번역, 요약, 의역
    *   **일반적인 크기**: 수백만 개의 매개변수

대규모 언어 모델은 다양한 형태로 제공되지만, LLM은 일반적으로 수십억 개의 매개변수를 가진 디코더 기반 모델입니다. 다음은 가장 잘 알려진 LLM 중 일부입니다.

| **모델** | **제공업체** |
|---|---|
| **Deepseek-R1** | DeepSeek |
| **GPT4** | OpenAI |
| **Llama 3** | Meta (Facebook AI Research) |
| **SmolLM2** | Hugging Face |
| **Gemma** | Google |
| **Mistral** | Mistral |

LLM의 기본 원리는 간단하면서도 매우 효과적입니다. 즉, 이전 토큰 시퀀스가 주어졌을 때 다음 토큰을 예측하는 것입니다. "토큰"은 LLM이 작업하는 정보의 단위입니다. "토큰"을 "단어"라고 생각할 수도 있지만, 효율성을 위해 LLM은 전체 단어를 사용하지 않습니다.

예를 들어, 영어에는 약 60만 개의 단어가 있지만, LLM은 약 32,000개의 토큰으로 구성된 어휘를 가질 수 있습니다(Llama 2의 경우). 토큰화는 종종 결합될 수 있는 하위 단어 단위로 작동합니다.

예를 들어, "interest"와 "ing" 토큰이 결합하여 "interesting"을 형성하거나, "ed"가 추가되어 "interested"를 형성하는 방식을 생각해 보세요.

아래 대화형 플레이그라운드에서 다양한 토크나이저를 실험해 볼 수 있습니다.

<iframe
	src="https://agents-course-the-tokenizer-playground.static.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

각 LLM에는 모델에 특정한 특수 토큰이 있습니다. LLM은 이러한 토큰을 사용하여 생성된 구조화된 구성 요소를 열고 닫습니다. 예를 들어, 시퀀스, 메시지 또는 응답의 시작 또는 끝을 나타내는 데 사용됩니다. 또한, 모델에 전달하는 입력 프롬프트도 특수 토큰으로 구성됩니다. 이 중 가장 중요한 것은 시퀀스 종료 토큰(EOS)입니다.

특수 토큰의 형태는 모델 제공업체마다 매우 다양합니다.

아래 표는 특수 토큰의 다양성을 보여줍니다.

<table>
  <thead>
    <tr>
      <th>**모델**</th>
      <th>**제공업체**</th>
      <th>**EOS 토큰**</th>
      <th>**기능**</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>**GPT4**</td>
      <td>OpenAI</td>
      <td><code>&lt;|endoftext|&gt;</code></td>
      <td>메시지 텍스트의 끝</td>
    </tr>
    <tr>
      <td>**Llama 3**</td>
      <td>Meta (Facebook AI Research)</td>
      <td><code>&lt;|eot_id|&gt;</code></td>
      <td>시퀀스 끝</td>
    </tr>
    <tr>
      <td>**Deepseek-R1**</td>
      <td>DeepSeek</td>
      <td><code>&lt;|end_of_sentence|&gt;</code></td>
      <td>메시지 텍스트의 끝</td>
    </tr>
    <tr>
      <td>**SmolLM2**</td>
      <td>Hugging Face</td>
      <td><code>&lt;|im_end|&gt;</code></td>
      <td>명령 또는 메시지의 끝</td>
    </tr>
    <tr>
      <td>**Gemma**</td>
      <td>Google</td>
      <td><code>&lt;end_of_turn&gt;</code></td>
      <td>대화 턴의 끝</td>
    </tr>
  </tbody>
</table>

<Tip>

이러한 특수 토큰을 모두 외울 필요는 없지만, LLM의 텍스트 생성에서 그 다양성과 역할의 중요성을 이해하는 것이 중요합니다. 특수 토큰에 대해 더 자세히 알고 싶다면, 모델의 허브 저장소에서 구성을 확인할 수 있습니다. 예를 들어, SmolLM2 모델의 특수 토큰은 <a href="https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/tokenizer_config.json">tokenizer_config.json</a>에서 찾을 수 있습니다.

</Tip>

## 다음 토큰 예측 이해하기.

LLM은 자기회귀적(autoregressive)이라고 합니다. 이는 한 번의 통과에서 나온 출력이 다음 통과의 입력이 된다는 의미입니다. 이 루프는 모델이 다음 토큰을 EOS 토큰으로 예측할 때까지 계속되며, 이 시점에서 모델은 멈출 수 있습니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/AutoregressionSchema.gif" alt="Visual Gif of autoregressive decoding" width="60%">

다시 말해, LLM은 EOS에 도달할 때까지 텍스트를 디코딩합니다. 하지만 단일 디코딩 루프 동안 어떤 일이 발생할까요?

에이전트 학습을 위한 전체 과정은 다소 기술적일 수 있지만, 간략한 개요는 다음과 같습니다.

*   입력 텍스트가 토큰화되면, 모델은 입력 시퀀스에서 각 토큰의 의미와 위치에 대한 정보를 포착하는 시퀀스 표현을 계산합니다.
*   이 표현은 모델로 들어가며, 모델은 어휘 내의 각 토큰이 시퀀스에서 다음 토큰이 될 가능성을 순위 매기는 점수를 출력합니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/DecodingFinal.gif" alt="Visual Gif of decoding" width="60%">

이러한 점수를 기반으로 문장을 완성할 토큰을 선택하는 여러 전략이 있습니다.

*   가장 쉬운 디코딩 전략은 항상 가장 높은 점수를 가진 토큰을 선택하는 것입니다.

이 Space에서 SmolLM2를 사용하여 디코딩 프로세스를 직접 상호 작용할 수 있습니다(이 모델의 **EOS** 토큰은 **<|im_end|>**이므로 해당 토큰에 도달할 때까지 디코딩됩니다).

<iframe
	src="https://agents-course-decoding-visualizer.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

*   하지만 더 고급 디코딩 전략도 있습니다. 예를 들어, *빔 서치(beam search)*는 일부 개별 토큰의 점수가 낮더라도 최대 총점을 가진 시퀀스를 찾기 위해 여러 후보 시퀀스를 탐색합니다.

<iframe
	src="https://agents-course-beam-search-visualizer.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

디코딩에 대해 더 자세히 알고 싶다면, [NLP 강좌](https://huggingface.co/learn/nlp-course)를 참조하세요.

## 어텐션(Attention)만 있으면 됩니다

트랜스포머 아키텍처의 핵심 측면은 어텐션(Attention)입니다. 다음 단어를 예측할 때, 문장의 모든 단어가 똑같이 중요하지는 않습니다. "프랑스의 수도는..."과 같은 문장에서 "프랑스"와 "수도"와 같은 단어가 가장 많은 의미를 전달합니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/AttentionSceneFinal.gif" alt="Visual Gif of Attention" width="60%">
다음 토큰을 예측하는 데 가장 관련성 높은 단어를 식별하는 이 과정은 매우 효과적인 것으로 입증되었습니다.

LLM의 기본 원리인 다음 토큰 예측은 GPT-2 이후로 일관되게 유지되었지만, 신경망 확장 및 어텐션 메커니즘이 점점 더 긴 시퀀스에서 작동하도록 하는 데 상당한 발전이 있었습니다.

LLM과 상호 작용해 본 적이 있다면, LLM이 처리할 수 있는 최대 토큰 수와 최대 *어텐션 범위*를 나타내는 *컨텍스트 길이*라는 용어에 익숙할 것입니다.

## LLM에 프롬프트하는 것이 중요합니다

LLM의 유일한 역할이 모든 입력 토큰을 보고 다음 토큰을 예측하고, 어떤 토큰이 "중요한지" 선택하는 것이라는 점을 고려할 때, 입력 시퀀스의 단어 선택은 매우 중요합니다.

LLM에 제공하는 입력 시퀀스를 *프롬프트*라고 합니다. 프롬프트를 신중하게 설계하면 LLM의 생성을 원하는 출력으로 더 쉽게 유도할 수 있습니다.

## LLM은 어떻게 훈련되나요?

LLM은 대규모 텍스트 데이터셋으로 훈련되며, 자기 지도 학습 또는 마스크 언어 모델링 목표를 통해 시퀀스에서 다음 단어를 예측하는 방법을 학습합니다.

이러한 비지도 학습을 통해 모델은 언어의 구조와 텍스트의 기본 패턴을 학습하여, 보지 못한 데이터에도 일반화할 수 있습니다.

이 초기 *사전 훈련* 후, LLM은 특정 작업을 수행하기 위해 지도 학습 목표에 따라 미세 조정될 수 있습니다. 예를 들어, 일부 모델은 대화 구조 또는 도구 사용을 위해 훈련되는 반면, 다른 모델은 분류 또는 코드 생성에 중점을 둡니다.

## LLM을 어떻게 사용할 수 있나요?

두 가지 주요 옵션이 있습니다.

1.  로컬에서 실행 (충분한 하드웨어가 있는 경우).
2.  클라우드/API 사용 (예: Hugging Face Serverless Inference API를 통해).

이 강좌에서는 주로 Hugging Face Hub의 API를 통해 모델을 사용할 것입니다. 나중에는 하드웨어에서 이러한 모델을 로컬로 실행하는 방법을 탐색할 것입니다.

## LLM은 AI 에이전트에서 어떻게 사용되나요?

LLM은 AI 에이전트의 핵심 구성 요소이며, 인간 언어를 이해하고 생성하는 기반을 제공합니다.

LLM은 사용자 지침을 해석하고, 대화의 컨텍스트를 유지하며, 계획을 정의하고, 사용할 도구를 결정할 수 있습니다.

이러한 단계는 이 유닛에서 더 자세히 탐색할 것이지만, 지금 이해해야 할 것은 LLM이 **에이전트의 두뇌**라는 것입니다.

---

많은 정보였습니다! LLM이 무엇인지, 어떻게 작동하는지, 그리고 AI 에이전트에 동력을 공급하는 역할에 대한 기본 사항을 다루었습니다.

언어 모델과 자연어 처리의 매혹적인 세계에 더 깊이 빠져들고 싶다면, 저희의 <a href="https://huggingface.co/learn/nlp-course/chapter1/1" target="_blank">무료 NLP 강좌</a>를 확인해 보세요.

이제 LLM이 어떻게 작동하는지 이해했으니, **LLM이 대화형 컨텍스트에서 생성을 어떻게 구성하는지** 알아볼 차례입니다.

<a href="https://huggingface.co/agents-course/notebooks/blob/main/unit1/dummy_agent_library.ipynb" target="_blank">이 노트북</a>을 실행하려면 <a href="https://hf.co/settings/tokens" target="_blank">https://hf.co/settings/tokens</a>에서 얻을 수 있는 **Hugging Face 토큰**이 필요합니다.

또한 <a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct" target="_blank">Meta Llama 모델</a>에 대한 접근 권한을 요청해야 합니다.