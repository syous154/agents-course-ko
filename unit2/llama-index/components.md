# LlamaIndex의 구성 요소는 무엇인가요?

1단원의 유용한 집사 에이전트 알프레드를 기억하시나요?
알프레드가 우리를 효과적으로 돕기 위해서는 우리의 요청을 이해하고 **관련 정보를 준비, 찾고 사용하여 작업을 완료하는 데 도움을 주어야 합니다.**
이것이 LlamaIndex의 구성 요소가 등장하는 지점입니다.

LlamaIndex에는 많은 구성 요소가 있지만, **특히 `QueryEngine` 구성 요소에 중점을 둘 것입니다.**
왜냐하면 에이전트를 위한 RAG(검색 증강 생성) 도구로 사용될 수 있기 때문입니다.

그렇다면 RAG는 무엇일까요? LLM은 일반 지식을 학습하기 위해 방대한 양의 데이터로 훈련됩니다.
그러나 관련성 있고 최신 데이터로 훈련되지 않을 수 있습니다.
RAG는 데이터에서 관련 정보를 찾아 검색하여 LLM에 제공함으로써 이 문제를 해결합니다.

![RAG](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/rag.png)

이제 알프레드가 어떻게 작동하는지 생각해 봅시다.

1.  저녁 식사 계획을 돕도록 알프레드에게 요청합니다.
2.  알프레드는 당신의 달력, 식단 선호도, 그리고 과거 성공적인 메뉴를 확인해야 합니다.
3.  `QueryEngine`은 알프레드가 이 정보를 찾고 저녁 식사 계획에 사용하는 데 도움을 줍니다.

이것은 `QueryEngine`을 LlamaIndex에서 **에이전트 RAG 워크플로우를 구축하는 핵심 구성 요소**로 만듭니다.
알프레드가 도움이 되기 위해 당신의 가정 정보를 검색해야 하는 것처럼, 모든 에이전트는 관련 데이터를 찾고 이해하는 방법이 필요합니다.
`QueryEngine`은 바로 이 기능을 제공합니다.

이제 구성 요소에 대해 좀 더 깊이 파고들어 **구성 요소를 결합하여 RAG 파이프라인을 만드는 방법**을 살펴보겠습니다.

## 구성 요소를 사용하여 RAG 파이프라인 생성

<Tip>
Google Colab을 사용하여 실행할 수 있는 <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/components.ipynb" target="_blank">이 노트북</a>에서 코드를 따라할 수 있습니다.
</Tip>

RAG 내에는 다섯 가지 주요 단계가 있으며, 이는 구축할 대부분의 대규모 애플리케이션의 일부가 될 것입니다. 이들은 다음과 같습니다.

1.  **로드**: 텍스트 파일, PDF, 다른 웹사이트, 데이터베이스 또는 API 등 데이터가 있는 곳에서 워크플로우로 데이터를 가져오는 것을 의미합니다. LlamaHub은 수백 가지의 통합 기능을 제공합니다.
2.  **인덱싱**: 데이터를 쿼리할 수 있는 데이터 구조를 생성하는 것을 의미합니다. LLM의 경우, 이는 거의 항상 벡터 임베딩을 생성하는 것을 의미합니다. 이는 데이터의 의미를 수치적으로 표현한 것입니다. 인덱싱은 속성을 기반으로 상황에 맞는 관련 데이터를 정확하게 찾는 것을 쉽게 하기 위한 수많은 다른 메타데이터 전략을 의미할 수도 있습니다.
3.  **저장**: 데이터가 인덱싱되면 다시 인덱싱할 필요가 없도록 인덱스 및 기타 메타데이터를 저장해야 합니다.
4.  **쿼리**: 주어진 인덱싱 전략에 대해 하위 쿼리, 다단계 쿼리 및 하이브리드 전략을 포함하여 LLM 및 LlamaIndex 데이터 구조를 활용하여 데이터를 쿼리하는 여러 가지 방법이 있습니다.
5.  **평가**: 모든 흐름에서 중요한 단계는 다른 전략과 비교하여 얼마나 효과적인지 또는 변경할 때 얼마나 효과적인지 확인하는 것입니다. 평가는 쿼리에 대한 응답이 얼마나 정확하고, 충실하며, 빠른지에 대한 객관적인 측정값을 제공합니다.

다음으로, 구성 요소를 사용하여 이러한 단계를 어떻게 재현할 수 있는지 살펴보겠습니다.

### 문서 로드 및 임베딩

앞서 언급했듯이, LlamaIndex는 자체 데이터 위에 작동할 수 있지만, **데이터에 액세스하기 전에 로드해야 합니다.**
LlamaIndex에 데이터를 로드하는 세 가지 주요 방법이 있습니다.

1.  `SimpleDirectoryReader`: 로컬 디렉토리에서 다양한 파일 유형을 위한 내장 로더.
2.  `LlamaParse`: LlamaIndex의 공식 PDF 파싱 도구인 LlamaParse는 관리형 API로 제공됩니다.
3.  `LlamaHub`: 모든 소스에서 데이터를 수집하기 위한 수백 개의 데이터 로딩 라이브러리 레지스트리.

<Tip>더 복잡한 데이터 소스의 경우 <a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/connector/">LlamaHub</a> 로더 및 <a href="https://github.com/run-llama/llama_cloud_services/blob/main/parse.md">LlamaParse</a> 파서에 익숙해지세요.</Tip>

**데이터를 로드하는 가장 간단한 방법은 `SimpleDirectoryReader`를 사용하는 것입니다.**
이 다재다능한 구성 요소는 폴더에서 다양한 파일 유형을 로드하고 LlamaIndex가 작업할 수 있는 `Document` 객체로 변환할 수 있습니다.
`SimpleDirectoryReader`를 사용하여 폴더에서 데이터를 로드하는 방법을 살펴보겠습니다.

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="path/to/directory")
documents = reader.load_data()
```

문서를 로드한 후, `Node` 객체라고 불리는 더 작은 조각으로 분할해야 합니다.
`Node`는 AI가 작업하기 더 쉬운 원본 문서의 텍스트 덩어리이며, 원본 `Document` 객체에 대한 참조를 여전히 가지고 있습니다.

`IngestionPipeline`은 두 가지 주요 변환을 통해 이러한 노드를 생성하는 데 도움을 줍니다.
1.  `SentenceSplitter`는 자연스러운 문장 경계에서 분할하여 문서를 관리 가능한 덩어리로 나눕니다.
2.  `HuggingFaceEmbedding`은 각 덩어리를 수치 임베딩으로 변환합니다. 이는 AI가 효율적으로 처리할 수 있는 방식으로 의미론적 의미를 포착하는 벡터 표현입니다.

이 프로세스는 검색 및 분석에 더 유용한 방식으로 문서를 구성하는 데 도움이 됩니다.

```python
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# 변환을 사용하여 파이프라인 생성
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)

nodes = await pipeline.arun(documents=[Document.example()])
```

### 문서 저장 및 인덱싱

`Node` 객체를 생성한 후 검색 가능하도록 인덱싱해야 하지만, 그렇게 하기 전에 데이터를 저장할 공간이 필요합니다.

수집 파이프라인을 사용하고 있으므로, 벡터 저장소를 파이프라인에 직접 연결하여 채울 수 있습니다.
이 경우, `Chroma`를 사용하여 문서를 저장할 것입니다.

<details>
<summary>ChromaDB 설치</summary>

[LlamaHub 섹션](./llama-hub)에서 소개된 바와 같이, 다음 명령으로 ChromaDB 벡터 저장소를 설치할 수 있습니다.

```bash
pip install llama-index-vector-stores-chroma
```
</details>

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)
```

<Tip>다양한 벡터 저장소에 대한 개요는 <a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/">LlamaIndex 문서</a>에서 찾을 수 있습니다.</Tip>

여기서 벡터 임베딩이 등장합니다. 쿼리와 노드를 동일한 벡터 공간에 임베딩함으로써 관련 일치를 찾을 수 있습니다.
`VectorStoreIndex`는 일관성을 보장하기 위해 수집 중에 사용한 것과 동일한 임베딩 모델을 사용하여 이 작업을 처리합니다.

벡터 저장소와 임베딩에서 이 인덱스를 생성하는 방법을 살펴보겠습니다.

```python
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
```

모든 정보는 `ChromaVectorStore` 객체와 전달된 디렉토리 경로 내에 자동으로 유지됩니다.

좋습니다! 이제 인덱스를 쉽게 저장하고 로드할 수 있으므로, 다양한 방법으로 쿼리하는 방법을 살펴보겠습니다.

### 프롬프트 및 LLM으로 VectorStoreIndex 쿼리

인덱스를 쿼리하기 전에 쿼리 인터페이스로 변환해야 합니다. 가장 일반적인 변환 옵션은 다음과 같습니다.

-   `as_retriever`: 기본 문서 검색용으로, 유사도 점수가 있는 `NodeWithScore` 객체 목록을 반환합니다.
-   `as_query_engine`: 단일 질문-답변 상호 작용용으로, 작성된 응답을 반환합니다.
-   `as_chat_engine`: 여러 메시지에서 메모리를 유지하는 대화형 상호 작용용으로, 채팅 기록 및 인덱싱된 컨텍스트를 사용하여 작성된 응답을 반환합니다.

에이전트와 유사한 상호 작용에 더 일반적이므로 쿼리 엔진에 중점을 둘 것입니다.
응답에 사용할 LLM도 쿼리 엔진에 전달합니다.

```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)
query_engine.query("인생의 의미는 무엇입니까?")
# 인생의 의미는 42입니다.
```

### 응답 처리

내부적으로 쿼리 엔진은 질문에 답하기 위해 LLM만 사용하는 것이 아니라, 응답을 처리하는 전략으로 `ResponseSynthesizer`도 사용합니다.
다시 한번, 이것은 완전히 사용자 정의할 수 있지만, 즉시 잘 작동하는 세 가지 주요 전략이 있습니다.

-   `refine`: 검색된 각 텍스트 덩어리를 순차적으로 거쳐 답변을 생성하고 다듬습니다. 이는 노드/검색된 덩어리당 별도의 LLM 호출을 만듭니다.
-   `compact` (기본값): 다듬기와 유사하지만, 덩어리를 미리 연결하여 LLM 호출 수를 줄입니다.
-   `tree_summarize`: 검색된 각 텍스트 덩어리를 거쳐 답변의 트리 구조를 생성하여 상세한 답변을 만듭니다.

<Tip><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/usage_pattern/#low-level-composition-api">저수준 구성 API</a>를 사용하여 쿼리 워크플로우를 세밀하게 제어하세요. 이 API를 사용하면 정확한 요구 사항에 맞게 쿼리 프로세스의 모든 단계를 사용자 지정하고 미세 조정할 수 있으며, 이는 <a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/">워크플로우</a>와도 잘 어울립니다.</Tip>

언어 모델이 항상 예측 가능한 방식으로 작동하지는 않으므로, 우리가 얻는 답변이 항상 정확하다고 확신할 수는 없습니다. 이는 **답변의 품질을 평가**하여 처리할 수 있습니다.

### 평가 및 관찰 가능성

LlamaIndex는 **응답 품질을 평가하기 위한 내장 평가 도구**를 제공합니다.
이러한 평가기는 LLM을 활용하여 다양한 차원에서 응답을 분석합니다.
사용 가능한 세 가지 주요 평가기를 살펴보겠습니다.

-   `FaithfulnessEvaluator`: 답변이 컨텍스트에 의해 뒷받침되는지 확인하여 답변의 충실도를 평가합니다.
-   `AnswerRelevancyEvaluator`: 답변이 질문과 관련이 있는지 확인하여 답변의 관련성을 평가합니다.
-   `CorrectnessEvaluator`: 답변이 올바른지 확인하여 답변의 정확성을 평가합니다.

<Tip>에이전트 관찰 가능성 및 평가에 대해 더 자세히 알고 싶으신가요? <a href="https://huggingface.co/learn/agents-course/bonus-unit2/introduction">보너스 유닛 2</a>에서 여정을 계속하세요.</Tip>

```python
from llama_index.core.evaluation import FaithfulnessEvaluator

query_engine = # 이전 섹션에서
llm = # 이전 섹션에서

# 인덱스 쿼리
evaluator = FaithfulnessEvaluator(llm=llm)
response = query_engine.query(
    "미국 독립 전쟁 중 뉴욕시에서 어떤 전투가 일어났습니까?"
)
eval_result = evaluator.evaluate_response(response=response)
eval_result.passing
```

직접적인 평가 없이도 **관찰 가능성을 통해 시스템이 어떻게 작동하는지에 대한 통찰력**을 얻을 수 있습니다.
이는 특히 더 복잡한 워크플로우를 구축하고 각 구성 요소가 어떻게 작동하는지 이해하고자 할 때 유용합니다.

<details>
<summary>LlamaTrace 설치</summary>

[LlamaHub 섹션](./llama-hub)에서 소개된 바와 같이, 다음 명령으로 Arize Phoenix의 LlamaTrace 콜백을 설치할 수 있습니다.

```bash
pip install -U llama-index-callbacks-arize-phoenix
```

또한, `PHOENIX_API_KEY` 환경 변수를 LlamaTrace API 키로 설정해야 합니다. 이는 다음을 통해 얻을 수 있습니다.
- [LlamaTrace](https://llamatrace.com/login)에서 계정 생성
- 계정 설정에서 API 키 생성
- 아래 코드에서 API 키를 사용하여 추적 활성화

</details>

```python
import llama_index
import os

PHOENIX_API_KEY = "<PHOENIX_API_KEY>"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
llama_index.core.set_global_handler(
    "arize_phoenix",
    endpoint="https://llamatrace.com/v1/traces"
)
```

<Tip>구성 요소 및 사용 방법에 대해 더 자세히 알고 싶으신가요? <a href="https://docs.llamaindex.ai/en/stable/module_guides/">구성 요소 가이드</a> 또는 <a href="https://docs.llamaindex.ai/en/stable/understanding/rag/">RAG 가이드</a>에서 여정을 계속하세요.</Tip>

구성 요소를 사용하여 `QueryEngine`을 생성하는 방법을 살펴보았습니다. 이제 **`QueryEngine`을 에이전트의 도구로 사용하는 방법**을 살펴보겠습니다!