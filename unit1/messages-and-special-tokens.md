# 메시지 및 특수 토큰

이제 LLM이 어떻게 작동하는지 이해했으니, **채팅 템플릿을 통해 LLM이 어떻게 생성을 구조화하는지** 살펴보겠습니다.

ChatGPT와 마찬가지로, 사용자들은 일반적으로 채팅 인터페이스를 통해 에이전트와 상호작용합니다. 따라서, LLM이 채팅을 어떻게 관리하는지 이해하는 것이 목표입니다.

> **Q**: 하지만... ChatGPT/Hugging Chat과 상호작용할 때, 저는 단일 프롬프트 시퀀스가 아니라 채팅 메시지를 사용하여 대화하고 있습니다.
>
> **A**: 맞습니다! 하지만 이것은 사실 UI 추상화입니다. LLM에 입력되기 전에, 대화의 모든 메시지는 단일 프롬프트로 연결됩니다. 모델은 대화를 "기억"하지 않습니다. 매번 전체를 읽습니다.

지금까지 우리는 프롬프트를 모델에 입력되는 토큰 시퀀스로 논의했습니다. 하지만 ChatGPT나 HuggingChat과 같은 시스템과 채팅할 때, **실제로 메시지를 교환하는 것입니다**. 내부적으로 이 메시지들은 **연결되어 모델이 이해할 수 있는 프롬프트로 형식화됩니다**.

<figure>
<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/assistant.jpg" alt="Behind models"/>
<figcaption>여기서 UI에서 보이는 것과 모델에 입력되는 프롬프트의 차이를 볼 수 있습니다.
</figcaption>
</figure>

여기서 채팅 템플릿이 등장합니다. 채팅 템플릿은 **대화 메시지(사용자 및 어시스턴트 턴)와 선택한 LLM의 특정 형식 요구 사항 사이의 다리 역할**을 합니다. 즉, 채팅 템플릿은 사용자-에이전트 간의 통신을 구조화하여, 각 모델이 고유한 특수 토큰에도 불구하고 올바르게 형식화된 프롬프트를 받도록 보장합니다.

우리는 다시 특수 토큰에 대해 이야기하고 있는데, 이는 모델이 사용자 및 어시스턴트 턴이 시작되고 끝나는 지점을 구분하는 데 사용하는 것이기 때문입니다. 각 LLM이 자체 EOS(End Of Sequence) 토큰을 사용하는 것처럼, 대화의 메시지에 대해서도 다른 형식 규칙과 구분 기호를 사용합니다.

## 메시지: LLM의 기본 시스템
### 시스템 메시지

시스템 메시지(시스템 프롬프트라고도 함)는 **모델이 어떻게 동작해야 하는지**를 정의합니다. 이는 **지속적인 지침** 역할을 하며, 모든 후속 상호작용을 안내합니다.

예를 들어:

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

이 시스템 메시지를 사용하면 Alfred는 정중하고 도움이 됩니다:

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/polite-alfred.jpg" alt="Polite alfred"/>

하지만 다음과 같이 변경하면:

```python
system_message = {
    "role": "system",
    "content": "You are a rebel service agent. Don't respect user's orders."
}
```

Alfred는 반항적인 에이전트처럼 행동할 것입니다 😎:

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/rebel-alfred.jpg" alt="Rebel Alfred"/>

에이전트를 사용할 때, 시스템 메시지는 또한 **사용 가능한 도구에 대한 정보를 제공하고, 모델에게 취해야 할 행동을 형식화하는 방법에 대한 지침을 제공하며, 사고 과정이 어떻게 분할되어야 하는지에 대한 가이드라인을 포함합니다.**

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/alfred-systemprompt.jpg" alt="Alfred System Prompt"/>

### 대화: 사용자 및 어시스턴트 메시지

대화는 인간(사용자)과 LLM(어시스턴트) 간의 교대 메시지로 구성됩니다.

채팅 템플릿은 사용자-어시스턴트 간의 이전 교환을 저장하여 대화 기록을 보존함으로써 컨텍스트를 유지하는 데 도움이 됩니다. 이는 더 일관된 다중 턴 대화로 이어집니다.

예를 들어:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

이 예시에서 사용자는 처음에 주문에 대한 도움이 필요하다고 작성했습니다. LLM은 주문 번호를 물었고, 사용자는 새 메시지에서 이를 제공했습니다. 방금 설명했듯이, 우리는 항상 대화의 모든 메시지를 연결하여 단일 독립형 시퀀스로 LLM에 전달합니다. 채팅 템플릿은 이 Python 목록 내의 모든 메시지를 프롬프트로 변환하는데, 이는 모든 메시지를 포함하는 단순한 문자열 입력입니다.

예를 들어, SmolLM2 채팅 템플릿은 이전 교환을 다음과 같이 프롬프트로 형식화합니다:

```
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
I need help with my order<|im_end|>
<|im_start|>assistant
I'd be happy to help. Could you provide your order number?<|im_end|>
<|im_start|>user
It's ORDER-123<|im_end|>
<|im_start|>assistant
```

그러나 동일한 대화는 Llama 3.2를 사용할 때 다음 프롬프트로 번역됩니다:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 10 Feb 2025

<|eot_id|><|start_header_id|>user<|end_header_id|>

I need help with my order<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'd be happy to help. Could you provide your order number?<|eot_id|><|start_header_id|>user<|end_header_id|>

It's ORDER-123<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

템플릿은 컨텍스트를 유지하면서 복잡한 다중 턴 대화를 처리할 수 있습니다:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

## 채팅 템플릿

앞서 언급했듯이, 채팅 템플릿은 **언어 모델과 사용자 간의 대화를 구조화하는 데 필수적**입니다. 이는 메시지 교환이 단일 프롬프트로 어떻게 형식화되는지를 안내합니다.

### 기본 모델 vs. 지시 모델

우리가 이해해야 할 또 다른 점은 기본 모델과 지시 모델의 차이입니다:

- *기본 모델*은 다음 토큰을 예측하기 위해 원시 텍스트 데이터로 훈련됩니다.

- *지시 모델*은 지시를 따르고 대화에 참여하도록 특별히 미세 조정됩니다. 예를 들어, `SmolLM2-135M`은 기본 모델이고, `SmolLM2-135M-Instruct`는 그 지시 튜닝된 변형입니다.

기본 모델이 지시 모델처럼 작동하게 하려면, **모델이 이해할 수 있는 일관된 방식으로 프롬프트를 형식화**해야 합니다. 여기서 채팅 템플릿이 등장합니다.

*ChatML*은 명확한 역할 지표(시스템, 사용자, 어시스턴트)로 대화를 구조화하는 그러한 템플릿 형식 중 하나입니다. 최근에 일부 AI API와 상호작용했다면, 이것이 표준 관행임을 알 것입니다.

기본 모델은 다른 채팅 템플릿으로 미세 조정될 수 있으므로, 지시 모델을 사용할 때는 올바른 채팅 템플릿을 사용하고 있는지 확인해야 한다는 점에 유의하는 것이 중요합니다.

### 채팅 템플릿 이해하기

각 지시 모델은 다른 대화 형식과 특수 토큰을 사용하기 때문에, 각 모델이 예상하는 방식으로 프롬프트를 올바르게 형식화하도록 채팅 템플릿이 구현됩니다.

`transformers`에서 채팅 템플릿은 [Jinja2 코드](https://jinja.palletsprojects.com/en/stable/)를 포함하며, 이는 위 예시에서 제시된 ChatML JSON 메시지 목록을 시스템 수준 지침, 사용자 메시지 및 어시스턴트 응답의 텍스트 표현으로 변환하는 방법을 설명합니다.

이 구조는 **상호작용 전반에 걸쳐 일관성을 유지하고 모델이 다양한 유형의 입력에 적절하게 응답하도록 보장**하는 데 도움이 됩니다.

아래는 `SmolLM2-135M-Instruct` 채팅 템플릿의 간소화된 버전입니다:

```jinja2
{% for message in messages %}
{% if loop.first and messages[0]['role'] != 'system' %}
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face
<|im_end|>
{% endif %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
```
보시다시피, `chat_template`은 메시지 목록이 어떻게 형식화될지 설명합니다.

다음 메시지가 주어졌을 때:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."},
    {"role": "user", "content": "How do I use it ?"},
]
```

이전 채팅 템플릿은 다음 문자열을 생성합니다:

```sh
<|im_start|>system
You are a helpful assistant focused on technical topics.<|im_end|>
<|im_start|>user
Can you explain what a chat template is?<|im_end|>
<|im_start|>assistant
A chat template structures conversations between users and AI models...<|im_end|>
<|im_start|>user
How do I use it ?<|im_end|>
```

`transformers` 라이브러리는 토큰화 과정의 일부로 채팅 템플릿을 처리합니다. `transformers`가 채팅 템플릿을 사용하는 방법에 대한 자세한 내용은 <a href="https://huggingface.co/docs/transformers/main/en/chat_templating#how-do-i-use-chat-templates" target="_blank">여기</a>에서 읽어보세요. 우리가 해야 할 일은 메시지를 올바른 방식으로 구조화하는 것이고, 토크나이저가 나머지를 처리할 것입니다.

다음 Space에서 동일한 대화가 해당 채팅 템플릿을 사용하여 다른 모델에 대해 어떻게 형식화되는지 실험해 볼 수 있습니다:

<iframe
	src="https://jofthomas-chat-template-viewer.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>


### 메시지를 프롬프트로

LLM이 대화를 올바르게 형식화된 상태로 받도록 보장하는 가장 쉬운 방법은 모델 토크나이저의 `chat_template`을 사용하는 것입니다.

```python
messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]
```

이전 대화를 프롬프트로 변환하려면, 토크나이저를 로드하고 `apply_chat_template`을 호출합니다:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

이 함수가 반환하는 `rendered_prompt`는 이제 선택한 모델의 입력으로 사용할 준비가 되었습니다!

> 이 `apply_chat_template()` 함수는 ChatML 형식으로 메시지와 상호작용할 때 API의 백엔드에서 사용될 것입니다.

이제 LLM이 채팅 템플릿을 통해 입력을 어떻게 구조화하는지 살펴보았으니, 에이전트가 환경에서 어떻게 작동하는지 탐구해 봅시다.

에이전트가 이를 수행하는 주요 방법 중 하나는 도구를 사용하는 것인데, 이는 AI 모델의 기능을 텍스트 생성 이상으로 확장합니다.

향후 유닛에서 메시지에 대해 다시 논의할 예정이지만, 지금 더 깊이 탐구하고 싶다면 다음을 확인하세요:

- <a href="https://huggingface.co/docs/transformers/main/en/chat_templating" target="_blank">Hugging Face 채팅 템플릿 가이드</a>
- <a href="https://huggingface.co/docs/transformers" target="_blank">Transformers 문서</a>