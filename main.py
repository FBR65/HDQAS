import logging
from retriever import HybridRetriever
from summarizer import QdrantSummarizer

from openai import OpenAI

from __init__ import (
    BASE_URL,
    API_KEY,
    MODEL_NAME,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Answering:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.retriever = HybridRetriever(collection_name=collection_name)
        self.summarizer = QdrantSummarizer(collection_name=collection_name)
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    async def answer(
        self,
        question: str,
        use_type: str = "retriever",
        max_tokens: int = 4096,
        top_k: int = 10,
        use_graph: bool = True,
    ) -> str:
        try:
            if use_type == "retriever":
                # Use the retriever to find relevant documents

                results = await self.retriever.retrieve(
                    query=question, top_k=top_k, use_graph=use_graph
                )
                system_prompt = "Erzeuge ausschliesslich aus dem zugelieferten Kontext eine sachlich, professionelle Antwort in Deutsch."
                user_prompt = f"{results}"
                try:
                    response = self.client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.2,
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    raise e
            else:
                retrieved_texts = self.summarizer.retrieve_all_texts()
                # Summarize the retrieved documents
                summary = self.summarizer.summarize_texts(
                    retrieved_texts, max_tokens=max_tokens
                )
                result = f"\n--- Generierte Zusammenfassung ---\n {summary}\n------------------------------\n"
                return result
            # Retrieve relevant documents

        except Exception as e:
            raise e


if __name__ == "__main__":
    import asyncio

    # WICHTIG: Geben Sie hier den *exakten* Namen der Qdrant Collection an!
    target_collection_name = "sum_collection_COSINE"

    answer = Answering(collection_name=target_collection_name)

    question = "Was macht der Admiral?"
    use_type = "retriever"
    max_tokens = 4096
    top_k = 10
    use_graph = True

    async def main_async():
        result_texts = await answer.answer(
            question, use_type, max_tokens, top_k, use_graph
        )
        print(result_texts)

    asyncio.run(main_async())
