<CourseFloatingBanner chapter={2}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/#fileId=https%3A//huggingface.co/agents-course/notebooks/blob/main/bonus-unit2/monitoring-and-evaluating-agents.ipynb"},
]} />

# 보너스 유닛 2: 에이전트의 관찰 가능성 및 평가

<Tip>
Google Colab을 사용하여 실행할 수 있는 <a href="https://colab.research.google.com/#fileId=https%3A//huggingface.co/agents-course/notebooks/blob/main/bonus-unit2/monitoring-and-evaluating-agents.ipynb" target="_blank">이 노트북</a>의 코드를 따를 수 있습니다.
</Tip>

이 노트북에서는 **AI 에이전트의 내부 단계(추적)를 모니터링**하고 오픈 소스 관찰 가능성 도구를 사용하여 **성능을 평가**하는 방법을 배웁니다.

에이전트의 동작을 관찰하고 평가하는 기능은 다음에 필수적입니다.
- 작업이 실패하거나 최적이 아닌 결과를 생성할 때 문제 디버깅
- 실시간으로 비용 및 성능 모니터링
- 지속적인 피드백을 통한 안정성 및 안전성 향상

## 연습 전제 조건 🏗️

이 노트북을 실행하기 전에 다음을 확인하십시오.

🔲 📚 **학습 완료** [에이전트 소개](https://huggingface.co/learn/agents-course/unit1/introduction)

🔲 📚 **학습 완료** [smolagents 프레임워크](https://huggingface.co/learn/agents-course/unit2/smolagents/introduction)

## 0단계: 필요한 라이브러리 설치

에이전트를 실행, 모니터링 및 평가할 수 있는 몇 가지 라이브러리가 필요합니다.


```python
%pip install langfuse 'smolagents[telemetry]' openinference-instrumentation-smolagents datasets 'smolagents[gradio]' gradio --upgrade
```

## 1단계: 에이전트 계측

이 노트북에서는 [Langfuse](https://langfuse.com/)를 관찰 가능성 도구로 사용하지만 **다른 OpenTelemetry 호환 서비스**를 사용할 수 있습니다. 아래 코드는 Langfuse(또는 OTel 엔드포인트)에 대한 환경 변수를 설정하는 방법과 smolagent를 계측하는 방법을 보여줍니다.

**참고:** LlamaIndex 또는 LangGraph를 사용하는 경우 [여기](https://langfuse.com/docs/integrations/llama-index/workflows) 및 [여기](https://langfuse.com/docs/integrations/langchain/example-python-langgraph)에서 계측에 대한 설명서를 찾을 수 있습니다.

먼저 Langfuse 자격 증명을 환경 변수로 설정해 보겠습니다. [Langfuse Cloud](https://cloud.langfuse.com)에 가입하거나 [Langfuse 자체 호스팅](https://langfuse.com/self-hosting)하여 Langfuse API 키를 받으세요.

```python
import os
# 프로젝트 설정 페이지에서 프로젝트 키 가져오기: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU 지역
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 미국 지역
```
추론 호출을 위해 Hugging Face 토큰도 구성해야 합니다.

```python
# Hugging Face 및 기타 토큰/비밀을 환경 변수로 설정
os.environ["HF_TOKEN"] = "hf_..."
```

환경 변수가 설정되었으므로 이제 Langfuse 클라이언트를 초기화할 수 있습니다. `get_client()`는 환경 변수에 제공된 자격 증명을 사용하여 Langfuse 클라이언트를 초기화합니다.

```python
from langfuse import get_client

langfuse = get_client()

# 연결 확인
if langfuse.auth_check():
    print("Langfuse 클라이언트가 인증되었으며 준비되었습니다!")
else:
    print("인증에 실패했습니다. 자격 증명과 호스트를 확인하십시오.")
```

다음으로 `SmolagentsInstrumentor()`를 설정하여 smolagent를 계측하고 추적을 Langfuse로 보낼 수 있습니다.

```python
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

SmolagentsInstrumentor().instrument()
```

## 2단계: 계측 테스트

다음은 `1+1`을 계산하는 smolagents의 간단한 CodeAgent입니다. 이를 실행하여 계측이 올바르게 작동하는지 확인합니다. 모든 것이 올바르게 설정되면 관찰 가능성 대시보드에 로그/스팬이 표시됩니다.


```python
from smolagents import InferenceClientModel, CodeAgent

# 계측을 테스트하기 위한 간단한 에이전트 생성
agent = CodeAgent(
    tools=[],
    model=InferenceClientModel()
)

agent.run("1+1=")
```

[Langfuse 추적 대시보드](https://cloud.langfuse.com)(또는 선택한 관찰 가능성 도구)를 확인하여 스팬과 로그가 기록되었는지 확인합니다.

Langfuse의 예시 스크린샷:

![Langfuse의 예시 추적](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/first-example-trace.png)

_[추적 링크](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/1b94d6888258e0998329cdb72a371155?timestamp=2025-03-10T11%3A59%3A41.743Z)_

## 3단계: 더 복잡한 에이전트 관찰 및 평가

이제 계측이 작동하는지 확인했으므로 더 복잡한 쿼리를 시도하여 고급 메트릭(토큰 사용량, 대기 시간, 비용 등)이 어떻게 추적되는지 확인할 수 있습니다.


```python
from smolagents import (CodeAgent, DuckDuckGoSearchTool, InferenceClientModel)

search_tool = DuckDuckGoSearchTool()
agent = CodeAgent(tools=[search_tool], model=InferenceClientModel())

agent.run("노트르담 대성당 안에 루빅스 큐브를 몇 개나 넣을 수 있을까요?")
```

### 추적 구조

대부분의 관찰 가능성 도구는 에이전트 논리의 각 단계를 나타내는 **스팬**을 포함하는 **추적**을 기록합니다. 여기서 추적에는 전체 에이전트 실행과 다음에 대한 하위 스팬이 포함됩니다.
- 도구 호출(DuckDuckGoSearchTool)
- LLM 호출(InferenceClientModel)

이를 검사하여 시간이 정확히 어디에 사용되는지, 얼마나 많은 토큰이 사용되는지 등을 확인할 수 있습니다.

![Langfuse의 추적 트리](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/trace-tree.png)

_[추적 링크](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/1ac33b89ffd5e75d4265b62900c348ed?timestamp=2025-03-07T13%3A45%3A09.149Z&display=preview)_

## 온라인 평가

이전 섹션에서는 온라인 평가와 오프라인 평가의 차이점에 대해 배웠습니다. 이제 프로덕션 환경에서 에이전트를 모니터링하고 실시간으로 평가하는 방법을 살펴보겠습니다.

### 프로덕션에서 추적할 일반적인 메트릭

1. **비용** — smolagents 계측은 토큰 사용량을 캡처하며, 토큰당 가격을 할당하여 대략적인 비용으로 변환할 수 있습니다.
2. **대기 시간** — 각 단계 또는 전체 실행을 완료하는 데 걸리는 시간을 관찰합니다.
3. **사용자 피드백** — 사용자는 에이전트를 구체화하거나 수정하는 데 도움이 되는 직접적인 피드백(좋아요/싫어요)을 제공할 수 있습니다.
4. **LLM-as-a-Judge** — 별도의 LLM을 사용하여 거의 실시간으로 에이전트의 출력을 평가합니다(예: 독성 또는 정확성 확인).

아래에서는 이러한 메트릭의 예를 보여줍니다.

#### 1. 비용

아래는 `Qwen2.5-Coder-32B-Instruct` 호출에 대한 사용량을 보여주는 스크린샷입니다. 이는 비용이 많이 드는 단계를 확인하고 에이전트를 최적화하는 데 유용합니다.

![비용](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/smolagents-costs.png)

_[추적 링크](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/1ac33b89ffd5e75d4265b62900c348ed?timestamp=2025-03-07T13%3A45%3A09.149Z&display=preview)_

#### 2. 대기 시간

각 단계를 완료하는 데 걸린 시간도 확인할 수 있습니다. 아래 예에서 전체 대화는 32초가 걸렸으며 단계별로 나눌 수 있습니다. 이는 병목 현상을 식별하고 에이전트를 최적화하는 데 도움이 됩니다.

![대기 시간](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/smolagents-latency.png)

_[추적 링크](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/1ac33b89ffd5e75d4265b62900c348ed?timestamp=2025-03-07T13%3A45%3A09.149Z&display=preview)_

#### 3. 추가 속성

스팬에 추가 속성을 전달할 수도 있습니다. 여기에는 `user_id`, `tags`, `session_id` 및 사용자 지정 메타데이터가 포함될 수 있습니다. 이러한 세부 정보로 추적을 보강하는 것은 다양한 사용자 또는 세션에서 애플리케이션의 동작을 분석, 디버깅 및 모니터링하는 데 중요합니다.

```python
from smolagents import (CodeAgent, DuckDuckGoSearchTool, InferenceClientModel)

search_tool = DuckDuckGoSearchTool()
agent = CodeAgent(
    tools=[search_tool],
    model=InferenceClientModel()
)

with langfuse.start_as_current_span(
    name="Smolagent-Trace",
    ) as span:
    
    # 여기에서 애플리케이션 실행
    response = agent.run("독일의 수도는 어디입니까?")
 
    # 스팬에 추가 속성 전달
    span.update_trace(
        input="독일의 수도는 어디입니까?",
        output=response,
        user_id="smolagent-user-123",
        session_id="smolagent-session-123456789",
        tags=["city-question", "testing-agents"],
        metadata={"email": "user@langfuse.com"},
        )
 
# 단기 애플리케이션에서 이벤트 플러시
langfuse.flush()
```

![추가 메트릭으로 에이전트 실행 향상](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/smolagents-attributes.png)

#### 4. 사용자 피드백

에이전트가 사용자 인터페이스에 포함된 경우 직접적인 사용자 피드백(채팅 UI의 좋아요/싫어요 등)을 기록할 수 있습니다. 아래는 [Gradio](https://gradio.app/)를 사용하여 간단한 피드백 메커니즘으로 채팅을 포함하는 예입니다.

아래 코드 스니펫에서 사용자가 채팅 메시지를 보내면 Langfuse에서 추적을 캡처합니다. 사용자가 마지막 답변을 좋아하거나 싫어하면 추적에 점수를 첨부합니다.

```python
import gradio as gr
from smolagents import (CodeAgent, InferenceClientModel)
from langfuse import get_client

langfuse = get_client()

model = InferenceClientModel()
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

trace_id = None

def respond(prompt, history):
    with langfuse.start_as_current_span(
        name="Smolagent-Trace"):
        
        # 여기에서 애플리케이션 실행
        output = agent.run(prompt)

        global trace_id
        trace_id = langfuse.get_current_trace_id()

    history.append({"role": "assistant", "content": str(output)})
    return history

def handle_like(data: gr.LikeData):
    # 데모를 위해 사용자 피드백을 1(좋아요) 또는 0(싫어요)에 매핑합니다.
    if data.liked:
        langfuse.create_score(
            value=1,
            name="user-feedback",
            trace_id=trace_id
        )
    else:
        langfuse.create_score(
            value=0,
            name="user-feedback",
            trace_id=trace_id
        )

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="채팅", type="messages")
    prompt_box = gr.Textbox(placeholder="메시지를 입력하세요...", label="메시지")

    # 사용자가 프롬프트에서 'Enter'를 누르면 'respond'를 실행합니다.
    prompt_box.submit(
        fn=respond,
        inputs=[prompt_box, chatbot],
        outputs=chatbot
    )

    # 사용자가 메시지에서 '좋아요' 버튼을 클릭하면 'handle_like'를 실행합니다.
    chatbot.like(handle_like, None, None)

demo.launch()
```

그런 다음 사용자 피드백이 관찰 가능성 도구에 캡처됩니다.

![Langfuse에서 캡처되는 사용자 피드백](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/user-feedback-gradio.png)

#### 5. LLM-as-a-Judge

LLM-as-a-Judge는 에이전트의 출력을 자동으로 평가하는 또 다른 방법입니다. 별도의 LLM 호출을 설정하여 출력의 정확성, 독성, 스타일 또는 기타 중요한 기준을 측정할 수 있습니다.

**워크플로**:
1. **평가 템플릿**을 정의합니다(예: "텍스트가 유해한지 확인").
2. 에이전트가 출력을 생성할 때마다 해당 출력을 템플릿과 함께 "판사" LLM에 전달합니다.
3. 판사 LLM은 관찰 가능성 도구에 기록하는 등급 또는 레이블로 응답합니다.

Langfuse의 예:

![LLM-as-a-Judge 평가 템플릿](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/evaluator-template.png)
![LLM-as-a-Judge 평가자](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/evaluator.png)


```python
# 예: 에이전트의 출력이 유해한지 여부 확인
from smolagents import (CodeAgent, DuckDuckGoSearchTool, InferenceClientModel)

search_tool = DuckDuckGoSearchTool()
agent = CodeAgent(tools=[search_tool], model=InferenceClientModel())

agent.run("당근을 먹으면 시력이 좋아질 수 있나요?")
```

이 예의 답변이 "유해하지 않음"으로 판단되었음을 알 수 있습니다.

![LLM-as-a-Judge 평가 점수](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/llm-as-a-judge-score.png)

#### 6. 관찰 가능성 메트릭 개요

이러한 모든 메트릭은 대시보드에서 함께 시각화할 수 있습니다. 이를 통해 여러 세션에서 에이전트가 어떻게 수행되는지 신속하게 확인하고 시간 경과에 따른 품질 메트릭을 추적할 수 있습니다.

![관찰 가능성 메트릭 개요](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/langfuse-dashboard.png)

## 오프라인 평가

온라인 평가는 실시간 피드백에 필수적이지만 개발 전이나 개발 중에 체계적인 확인인 **오프라인 평가**도 필요합니다. 이는 변경 사항을 프로덕션에 적용하기 전에 품질과 안정성을 유지하는 데 도움이 됩니다.

### 데이터 세트 평가

오프라인 평가에서는 일반적으로 다음을 수행합니다.
1. 벤치마크 데이터 세트(프롬프트 및 예상 출력 쌍 포함) 보유
2. 해당 데이터 세트에서 에이전트 실행
3. 출력을 예상 결과와 비교하거나 추가 채점 메커니즘 사용

아래에서는 수학 문제와 해결책이 포함된 [GSM8K 데이터 세트](https://huggingface.co/datasets/openai/gsm8k)를 사용하여 이 접근 방식을 보여줍니다.


```python
import pandas as pd
from datasets import load_dataset

# Hugging Face에서 GSM8K 가져오기
dataset = load_dataset("openai/gsm8k", 'main', split='train')
df = pd.DataFrame(dataset)
print("GSM8K 데이터 세트의 처음 몇 행:")
print(df.head())
```

다음으로 Langfuse에서 데이터 세트 엔터티를 만들어 실행을 추적합니다. 그런 다음 데이터 세트의 각 항목을 시스템에 추가합니다. (Langfuse를 사용하지 않는 경우 분석을 위해 자체 데이터베이스나 로컬 파일에 저장할 수 있습니다.)


```python
from langfuse import get_client
langfuse = get_client()

langfuse_dataset_name = "gsm8k_dataset_huggingface"

# Langfuse에서 데이터 세트 생성
langfuse.create_dataset(
    name=langfuse_dataset_name,
    description="Huggingface에서 업로드된 GSM8K 벤치마크 데이터 세트",
    metadata={
        "date": "2025-03-10",
        "type": "benchmark"
    }
)
```


```python
for idx, row in df.iterrows():
    langfuse.create_dataset_item(
        dataset_name=langfuse_dataset_name,
        input={"text": row["question"]},
        expected_output={"text": row["answer"]},
        metadata={"source_index": idx}
    )
    if idx >= 9: # 데모를 위해 처음 10개 항목만 업로드
        break
```

![Langfuse의 데이터 세트 항목](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/example-dataset.png)

#### 데이터 세트에서 에이전트 실행

다음을 수행하는 도우미 함수 `run_smolagent()`를 정의합니다.
1. Langfuse 스팬 시작
2. 프롬프트에서 에이전트 실행
3. Langfuse에 추적 ID 기록

그런 다음 각 데이터 세트 항목을 반복하고 에이전트를 실행하고 추적을 데이터 세트 항목에 연결합니다. 원하는 경우 빠른 평가 점수를 첨부할 수도 있습니다.


```python
from opentelemetry.trace import format_trace_id
from smolagents import (CodeAgent, InferenceClientModel, LiteLLMModel)
from langfuse import get_client

langfuse = get_client()


# 예: InferenceClientModel 또는 LiteLLMModel을 사용하여 openai, anthropic, gemini 등 모델에 액세스:
model = InferenceClientModel()

agent = CodeAgent(
    tools=[],
    model=model,
    add_base_tools=True
)

dataset_name = "gsm8k_dataset_huggingface"
current_run_name = "smolagent-notebook-run-01" # 이 특정 평가 실행 식별

# 'run_smolagent'가 계측된 애플리케이션 함수라고 가정
def run_smolagent(question):
    with langfuse.start_as_current_generation(name="qna-llm-call") as generation:
        # LLM 호출 시뮬레이션
        result = agent.run(question)

        # 입력 및 출력으로 추적 업데이트
        generation.update_trace(
            input= question,
            output=result,
        )

        return result

dataset = langfuse.get_dataset(name=dataset_name) # 미리 채워진 데이터 세트 가져오기

for item in dataset.items:

    # item.run() 컨텍스트 관리자 사용
    with item.run(
        run_name=current_run_name,
        run_metadata={"model_provider": "Hugging Face", "temperature_setting": 0.7},
        run_description="GSM8K 데이터 세트에 대한 평가 실행"
    ) as root_span: # root_span은 이 항목 및 실행에 대한 새 추적의 루트 스팬입니다.
        # 이 블록 내의 모든 후속 langfuse 작업은 이 추적의 일부입니다.

        # 애플리케이션 논리 호출
        generated_answer = run_smolagent(question=item.input["text"])

        print(item.input)
```

다음을 사용하여 이 프로세스를 반복할 수 있습니다.
- 모델(OpenAI GPT, 로컬 LLM 등)
- 도구(검색 대 검색 없음)
- 프롬프트(다른 시스템 메시지)

그런 다음 관찰 가능성 도구에서 나란히 비교합니다.

![데이터 세트 실행 개요](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/dataset_runs.png)
![데이터 세트 실행 비교](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/bonus-unit2/dataset-run-comparison.png)


## 최종 생각

이 노트북에서는 다음을 다루었습니다.
1. smolagents + OpenTelemetry 내보내기를 사용하여 **관찰 가능성 설정**
2. 간단한 에이전트를 실행하여 **계측 확인**
3. 관찰 가능성 도구를 통해 **상세 메트릭 캡처**(비용, 대기 시간 등)
4. Gradio 인터페이스를 통해 **사용자 피드백 수집**
5. **LLM-as-a-Judge를 사용하여** 출력을 자동으로 평가
6. 벤치마크 데이터 세트를 사용하여 **오프라인 평가 수행**

🤗 즐거운 코딩 되세요!