<CourseFloatingBanner
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/#fileId=https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/retrieval_agents.ipynb"},
]}
askForHelpUrl="http://hf.co/join/discord" />

# 에이전트 기반 RAG 시스템 구축하기

<Tip>
Google Colab을 사용하여 실행할 수 있는 <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/retrieval_agents.ipynb" target="_blank">이 노트북</a>의 코드를 따라할 수 있습니다.
</Tip>

검색 증강 생성(RAG) 시스템은 데이터 검색 및 생성 모델의 기능을 결합하여 문맥을 인지하는 응답을 제공합니다. 예를 들어, 사용자의 쿼리가 검색 엔진으로 전달되고, 검색된 결과는 쿼리와 함께 모델에 제공됩니다. 그러면 모델은 쿼리 및 검색된 정보를 기반으로 응답을 생성합니다.

에이전트 기반 RAG(Agentic RAG)는 **자율 에이전트와 동적 지식 검색을 결합**하여 기존 RAG 시스템을 확장합니다.

기존 RAG 시스템이 검색된 데이터를 기반으로 LLM이 쿼리에 답변하도록 하는 반면, 에이전트 기반 RAG는 **검색 및 생성 프로세스 모두에 대한 지능적인 제어를 가능하게 하여** 효율성과 정확성을 향상시킵니다.

기존 RAG 시스템은 **단일 검색 단계에 의존**하고 사용자 쿼리와의 직접적인 의미론적 유사성에 초점을 맞추는 등 주요 한계에 직면하며, 이로 인해 관련 정보를 놓칠 수 있습니다.

에이전트 기반 RAG는 에이전트가 검색 쿼리를 자율적으로 구성하고, 검색된 결과를 비판적으로 평가하며, 보다 맞춤화되고 포괄적인 출력을 위해 여러 검색 단계를 수행하도록 허용함으로써 이러한 문제를 해결합니다.

## DuckDuckGo를 사용한 기본 검색

DuckDuckGo를 사용하여 웹을 검색할 수 있는 간단한 에이전트를 만들어 보겠습니다. 이 에이전트는 정보를 검색하고 응답을 종합하여 쿼리에 답변합니다. 에이전트 기반 RAG를 통해 Alfred의 에이전트는 다음을 수행할 수 있습니다.

* 최신 슈퍼히어로 파티 트렌드 검색
* 고급 요소를 포함하도록 결과 정제
* 정보를 종합하여 완전한 계획 수립

Alfred의 에이전트가 이를 달성하는 방법은 다음과 같습니다.

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

# Initialize the search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the model
model = InferenceClientModel()

agent = CodeAgent(
    model=model,
    tools=[search_tool],
)

# Example usage
response = agent.run(
    "Search for luxury superhero-themed party ideas, including decorations, entertainment, and catering."
)
print(response)
```

에이전트는 다음 프로세스를 따릅니다.

1.  **요청 분석:** Alfred의 에이전트는 쿼리의 핵심 요소(고급 슈퍼히어로 테마 파티 계획, 장식, 엔터테인먼트, 케이터링에 중점)를 식별합니다.
2.  **검색 수행:** 에이전트는 DuckDuckGo를 활용하여 가장 관련성 높고 최신 정보를 검색하여 Alfred의 고급 이벤트에 대한 정제된 선호도에 부합하는지 확인합니다.
3.  **정보 종합:** 결과를 수집한 후 에이전트는 Alfred를 위한 응집력 있고 실행 가능한 계획으로 정보를 처리하여 파티의 모든 측면을 다룹니다.
4.  **향후 참조를 위해 저장:** 에이전트는 검색된 정보를 저장하여 향후 이벤트를 계획할 때 쉽게 액세스할 수 있도록 하여 후속 작업의 효율성을 최적화합니다.

## 사용자 지정 지식 기반 도구

특수 작업을 위해서는 사용자 지정 지식 기반이 매우 중요할 수 있습니다. 기술 문서 또는 특수 지식의 벡터 데이터베이스를 쿼리하는 도구를 만들어 보겠습니다. 의미론적 검색을 사용하여 에이전트는 Alfred의 요구 사항에 가장 관련성 높은 정보를 찾을 수 있습니다.

벡터 데이터베이스는 기계 학습 모델에 의해 생성된 텍스트 또는 기타 데이터의 숫자 표현(임베딩)을 저장합니다. 이는 고차원 공간에서 유사한 의미를 식별하여 의미론적 검색을 가능하게 합니다.

이 접근 방식은 사전 정의된 지식과 의미론적 검색을 결합하여 이벤트 계획을 위한 문맥을 인지하는 솔루션을 제공합니다. 특수 지식 액세스를 통해 Alfred는 파티의 모든 세부 사항을 완벽하게 만들 수 있습니다.

이 예에서는 사용자 지정 지식 기반에서 파티 계획 아이디어를 검색하는 도구를 만듭니다. 지식 기반을 검색하고 상위 결과를 반환하기 위해 BM25 검색기를 사용하고, 더 효율적인 검색을 위해 문서를 더 작은 청크로 분할하기 위해 `RecursiveCharacterTextSplitter`를 사용합니다.

```python
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, InferenceClientModel

class PartyPlanningRetrieverTool(Tool):
    name = "party_planning_retriever"
    description = "Uses semantic search to retrieve relevant party planning ideas for Alfred’s superhero-themed party at Wayne Manor."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be a query related to party planning or superhero themes.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=5  # Retrieve the top 5 documents
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved ideas:\n" + "".join(
            [
                f"\n\n===== Idea {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Simulate a knowledge base about party planning
party_ideas = [
    {"text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.", "source": "Party Ideas 1"},
    {"text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.", "source": "Entertainment Ideas"},
    {"text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'", "source": "Catering Ideas"},
    {"text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.", "source": "Decoration Ideas"},
    {"text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.", "source": "Entertainment Ideas"}
]

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in party_ideas
]

# Split the documents into smaller chunks for more efficient search
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""]
)
docs_processed = text_splitter.split_documents(source_docs)

# Create the retriever tool
party_planning_retriever = PartyPlanningRetrieverTool(docs_processed)

# Initialize the agent
agent = CodeAgent(tools=[party_planning_retriever], model=InferenceClientModel())

# Example usage
response = agent.run(
    "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options."
)

print(response)
```

이 향상된 에이전트는 다음을 수행할 수 있습니다.
1.  먼저 관련 정보에 대한 문서를 확인합니다.
2.  지식 기반의 통찰력을 결합합니다.
3.  대화 문맥을 메모리에 유지합니다.

## 향상된 검색 기능

에이전트 기반 RAG 시스템을 구축할 때 에이전트는 다음과 같은 정교한 전략을 사용할 수 있습니다.

1.  **쿼리 재구성(Query Reformulation):** 원시 사용자 쿼리 대신, 에이전트는 대상 문서와 더 잘 일치하는 최적화된 검색어를 작성할 수 있습니다.
2.  **쿼리 분해(Query Decomposition):** 사용자 쿼리에 쿼리할 정보가 여러 개 포함된 경우, 사용자 쿼리를 직접 사용하는 대신 여러 쿼리로 분해할 수 있습니다.
3.  **쿼리 확장(Query Expansion):** 쿼리 재구성과 다소 유사하지만, 여러 번 수행하여 쿼리를 여러 단어로 표현하여 모두 쿼리합니다.
4.  **재순위화(Reranking):** 검색된 문서와 검색 쿼리 간에 더 포괄적이고 의미론적인 관련성 점수를 할당하기 위해 크로스 인코더를 사용합니다.
5.  **다단계 검색(Multi-Step Retrieval):** 에이전트는 여러 검색을 수행하여 초기 결과를 사용하여 후속 쿼리에 정보를 제공할 수 있습니다.
6.  **소스 통합(Source Integration):** 웹 검색 및 로컬 문서와 같은 여러 소스의 정보를 결합할 수 있습니다.
7.  **결과 유효성 검사(Result Validation):** 검색된 콘텐츠는 응답에 포함되기 전에 관련성 및 정확성에 대해 분석될 수 있습니다.

효과적인 에이전트 기반 RAG 시스템은 몇 가지 주요 측면을 신중하게 고려해야 합니다. 에이전트는 **쿼리 유형 및 문맥에 따라 사용 가능한 도구 중에서 선택**해야 합니다. 메모리 시스템은 대화 기록을 유지하고 반복적인 검색을 방지하는 데 도움이 됩니다. 대체 전략을 사용하면 기본 검색 방법이 실패하더라도 시스템이 여전히 가치를 제공할 수 있습니다. 또한 유효성 검사 단계를 구현하면 검색된 정보의 정확성과 관련성을 보장하는 데 도움이 됩니다.

## 자료

- [Agentic RAG: 쿼리 재구성 및 자체 쿼리로 RAG를 터보차지하세요! 🚀](https://huggingface.co/learn/cookbook/agent_rag) - smolagents를 사용하여 에이전트 기반 RAG 시스템을 개발하기 위한 레시피.
