import logging
from typing import List

from openai import OpenAI, RateLimitError, APIError

from qdrant_client import QdrantClient

from __init__ import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
    BASE_URL,
    API_KEY,
    SUMMARIZATION_MODEL,
    MODEL_CONTEXT_LIMIT_TOKENS,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: Import tiktoken für genauere Zählung
try:
    import tiktoken

    tokenizer = tiktoken.get_encoding("cl100k_base")  # Passen Sie ggf. das Encoding an
    logger.info("Tiktoken-Tokenizer geladen.")
except ImportError:
    tokenizer = None
    logger.warning("Tiktoken nicht gefunden. Token-Schätzung basiert auf Zeichenlänge.")


class QdrantSummarizer:
    """
    A class for retrieving ALL texts from a Qdrant collection and summarizing them,
    using direct or map-reduce strategy based on estimated token count.
    """

    MAP_REDUCE_THRESHOLD_PERCENT = 0.85
    APPROX_TOKENS_PER_CHAR = 1 / 3.5

    def __init__(self, collection_name: str) -> None:
        """
        Initialize the QdrantSummarizer object.

        Args:
            collection_name (str): The exact name of the Qdrant collection.
        """
        self.collection_name = collection_name
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.summarization_model = SUMMARIZATION_MODEL
        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT
        )
        self.token_threshold = int(
            MODEL_CONTEXT_LIMIT_TOKENS * self.MAP_REDUCE_THRESHOLD_PERCENT
        )

        logger.info(f"Verbunden mit Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        logger.info(f"Verwende Collection: {self.collection_name}")
        logger.info(
            f"Verwende Summarization Model: {self.summarization_model} via {BASE_URL}"
        )
        logger.info(f"Bekanntes Kontextlimit: {MODEL_CONTEXT_LIMIT_TOKENS} Tokens")
        logger.info(
            f"Map-Reduce Schwellenwert: {self.token_threshold} Tokens ({self.MAP_REDUCE_THRESHOLD_PERCENT * 100}%)"
        )
        if not tokenizer:
            logger.warning(
                "Tiktoken nicht verfügbar, verwende Zeichen-basierte Token-Schätzung."
            )
        # Prüfen, ob die Collection existiert und Infos loggen
        try:
            collection_info = self.qdrant_client.get_collection(
                collection_name=self.collection_name
            )
            logger.info(
                f"Collection '{self.collection_name}' gefunden. Enthält ca. {collection_info.points_count} Punkte."
            )
        except Exception as e:
            logger.error(
                f"Fehler beim Abrufen der Collection-Info für '{self.collection_name}': {e}",
                exc_info=True,
            )
            # Hier könnte man entscheiden, ob man abbricht oder weitermacht
            # raise ValueError(f"Collection '{self.collection_name}' nicht gefunden oder Qdrant nicht erreichbar.") from e

    def _estimate_tokens(self, text: str) -> int:
        """Schätzt die Anzahl der Tokens für einen gegebenen Text."""
        # (Unverändert)
        if tokenizer:
            try:
                return len(tokenizer.encode(text))
            except Exception as e:
                logger.warning(
                    f"Fehler bei Tiktoken-Encoding, falle zurück auf Zeichenzählung: {e}"
                )
                return int(len(text) * self.APPROX_TOKENS_PER_CHAR)
        else:
            return int(len(text) * self.APPROX_TOKENS_PER_CHAR)

    def retrieve_all_texts(self) -> List[str]:  # Kein 'limit' Parameter mehr
        """
        Retrieves ALL texts from the specified Qdrant collection.
        """
        texts = []
        next_page_offset = None
        retrieved_count = 0
        processed_ids = set()

        try:
            # Gesamtzahl der Punkte für Logging abrufen
            try:
                # Verwende count für eine möglicherweise schnellere Abfrage der Anzahl
                count_result = self.qdrant_client.count(
                    collection_name=self.collection_name, exact=False
                )  # exact=False ist schneller
                total_points = count_result.count
                logger.info(
                    f"Rufe alle {total_points} (geschätzt) Dokumente aus Collection '{self.collection_name}' ab..."
                )
            except Exception as count_exc:
                logger.warning(
                    f"Konnte genaue Anzahl nicht schnell ermitteln ({count_exc}). Fahre mit Scroll fort."
                )
                logger.info(
                    f"Rufe alle Dokumente aus Collection '{self.collection_name}' ab..."
                )

            while True:
                # Feste Batchgröße für Scroll
                current_batch_size = 1000
                logger.debug(
                    f"Scrolling Qdrant: batch_size={current_batch_size}, offset={next_page_offset}"
                )

                # Scroll API verwenden, um Dokumente zu holen
                scroll_response, next_page_offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=current_batch_size,  # Verarbeite in Batches
                    offset=next_page_offset,
                    with_payload=True,  # Nur Payload nötig
                    with_vectors=False,
                )

                if not scroll_response:
                    logger.debug("Keine weiteren Dokumente im Scroll-Ergebnis.")
                    break  # Keine weiteren Dokumente mehr

                new_records_in_batch = 0
                for record in scroll_response:
                    # Doppelte Einträge vermeiden, falls Offset-Logik nicht perfekt ist
                    if record.id not in processed_ids:
                        processed_ids.add(record.id)
                        if record.payload and "text" in record.payload:
                            texts.append(record.payload["text"])
                            retrieved_count += 1
                            new_records_in_batch += 1
                        else:
                            logger.warning(
                                f"Record {record.id} hat keinen 'text' im Payload."
                            )

                logger.debug(
                    f"{new_records_in_batch} neue Dokumente in diesem Batch verarbeitet (Gesamt: {retrieved_count})."
                )

                # Wenn kein Offset mehr zurückkommt, sind wir fertig
                if not next_page_offset:
                    logger.debug(
                        "Kein next_page_offset von Qdrant erhalten, beende Scroll."
                    )
                    break

            logger.info(
                f"Insgesamt {len(texts)} Texte ({retrieved_count} Dokumente) aus der Collection abgerufen."
            )
            return texts

        except Exception as e:
            logger.error(
                f"Fehler beim Abrufen von Dokumenten aus Qdrant: {e}", exc_info=True
            )
            # Gib zurück, was bisher gesammelt wurde, auch bei Fehlern
            return texts

    def _summarize_chunk(
        self, chunk_text: str, max_tokens: int, attempt: int = 1
    ) -> str:
        """Helper-Funktion zum Zusammenfassen eines einzelnen Text-Chunks."""
        # (Unverändert)
        system_prompt = "Fasse den folgenden Text prägnant und objektiv zusammen."
        user_prompt = f"Text:\n{chunk_text}\n\nZusammenfassung:"
        try:
            response = self.client.chat.completions.create(
                model=self.summarization_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            logger.warning(f"Rate Limit erreicht bei Versuch {attempt}. Warte... ({e})")
            if attempt < 3:
                import time

                time.sleep(5 * attempt)
                return self._summarize_chunk(chunk_text, max_tokens, attempt + 1)
            else:
                logger.error(f"Rate Limit nach {attempt} Versuchen. Breche ab.")
                raise e
        except APIError as e:
            if "context_length_exceeded" in str(e).lower():
                logger.error(f"Kontextlängenfehler bei Chunk: {e}.")
                return f"[Fehler: Chunk zu lang] {chunk_text[:100]}..."
            else:
                logger.error(f"API Fehler (Versuch {attempt}): {e}")
                raise e
        except Exception as e:
            logger.error(f"Allg. Fehler (Versuch {attempt}): {e}", exc_info=True)
            return f"[Fehler bei Zusammenfassung] {chunk_text[:100]}..."

    def _map_reduce_summarize(self, texts: List[str], max_tokens_final: int) -> str:
        """Führt die Map-Reduce-Zusammenfassung durch."""
        # (Unverändert)
        logger.info("Starte Map-Reduce-Zusammenfassung...")
        intermediate_summaries = []
        current_chunk = []
        current_chunk_tokens = 0
        max_tokens_intermediate = max(100, max_tokens_final // 2)
        chunk_token_limit = self.token_threshold - 500
        separator = "\n\n---\n\n"
        separator_tokens = self._estimate_tokens(separator)

        for i, text in enumerate(texts):
            text_tokens = self._estimate_tokens(text)
            if (
                current_chunk_tokens
                + text_tokens
                + (separator_tokens if current_chunk else 0)
                <= chunk_token_limit
            ):
                current_chunk.append(text)
                current_chunk_tokens += text_tokens + (
                    separator_tokens if len(current_chunk) > 1 else 0
                )
            else:
                if current_chunk:
                    logger.info(
                        f"Fasse Chunk {len(intermediate_summaries) + 1} zusammen ({current_chunk_tokens} Tokens)..."
                    )
                    chunk_text_joined = separator.join(current_chunk)
                    summary = self._summarize_chunk(
                        chunk_text_joined, max_tokens_intermediate
                    )
                    intermediate_summaries.append(summary)
                if text_tokens <= chunk_token_limit:
                    current_chunk = [text]
                    current_chunk_tokens = text_tokens
                else:
                    logger.warning(
                        f"Dokument {i + 1} ({text_tokens} Tokens) überschreitet Chunk-Limit ({chunk_token_limit}) -> übersprungen."
                    )
                    current_chunk = []
                    current_chunk_tokens = 0
        if current_chunk:
            logger.info(
                f"Fasse letzten Chunk {len(intermediate_summaries) + 1} zusammen ({current_chunk_tokens} Tokens)..."
            )
            chunk_text_joined = separator.join(current_chunk)
            summary = self._summarize_chunk(chunk_text_joined, max_tokens_intermediate)
            intermediate_summaries.append(summary)

        logger.info(
            f"Map-Phase: {len(intermediate_summaries)} Zwischenzusammenfassungen."
        )
        if not intermediate_summaries:
            return "Keine Zusammenfassungen erstellt."
        if len(intermediate_summaries) == 1:
            return intermediate_summaries[0]

        logger.info("Starte Reduce-Phase...")
        combined_summaries_text = separator.join(intermediate_summaries)
        combined_tokens = self._estimate_tokens(combined_summaries_text)
        if combined_tokens > self.token_threshold:
            logger.warning(
                f"Kombinierte Summaries ({combined_tokens}) > Limit ({self.token_threshold}). Reduziere nur erste Stufe."
            )
            # Hier könnte man eine tiefere Rekursion einbauen, wenn nötig
        logger.info(
            f"Fasse {len(intermediate_summaries)} Zwischenzusammenfassungen ({combined_tokens} Tokens) final zusammen..."
        )
        final_summary = self._summarize_chunk(combined_summaries_text, max_tokens_final)
        logger.info("Map-Reduce-Zusammenfassung abgeschlossen.")
        return final_summary

    def summarize_texts(self, texts: List[str], max_tokens: int = 500) -> str:
        """
        Summarizes a list of texts, choosing between direct and map-reduce strategy.
        """
        # (Unverändert)
        if not texts:
            return "Kein Text zur Zusammenfassung vorhanden."
        full_text = "\n\n---\n\n".join(texts)
        estimated_total_tokens = self._estimate_tokens(full_text)
        logger.info(
            f"Geschätzte Gesamt-Tokens für {len(texts)} Dokumente: {estimated_total_tokens}"
        )

        if estimated_total_tokens <= self.token_threshold:
            logger.info("Versuche direkte Zusammenfassung...")
            try:
                system_prompt = "Du bist ein Assistent. Fasse die folgenden Textabschnitte prägnant zusammen."
                user_prompt = f"Textabschnitte:\n--- START ---\n{full_text}\n--- ENDE ---\n\nZusammenfassung:"
                response = self.client.chat.completions.create(
                    model=self.summarization_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                summary = response.choices[0].message.content.strip()
                logger.info("Direkte Zusammenfassung erfolgreich.")
                return summary
            except APIError as e:
                if "context_length_exceeded" in str(e).lower():
                    logger.warning(
                        f"Direkte Zusammenfassung fehlgeschlagen (Kontextlänge). Wechsle zu Map-Reduce. Fehler: {e}"
                    )
                    return self._map_reduce_summarize(texts, max_tokens)
                else:
                    logger.error(f"API Fehler bei direkter Zusammenfassung: {e}")
                    return f"Fehler: {e}"
            except Exception as e:
                logger.error(
                    f"Allg. Fehler bei direkter Zusammenfassung: {e}", exc_info=True
                )
                return f"Fehler: {e}"
        else:
            logger.info("Tokens > Schwellenwert. Verwende Map-Reduce.")
            return self._map_reduce_summarize(texts, max_tokens)


if __name__ == "__main__":
    # WICHTIG: Geben Sie hier den *exakten* Namen der Qdrant Collection an!
    target_collection_name = "sum_collection_COSINE"  # Beispiel, bitte anpassen!

    summarizer = QdrantSummarizer(collection_name=target_collection_name)

    # Kein doc_limit mehr nötig

    # 1. ALLE Texte aus Qdrant abrufen
    print(f"\nRufe ALLE Texte aus Collection '{target_collection_name}' ab...")
    retrieved_texts = summarizer.retrieve_all_texts()  # Aufruf ohne Limit

    # 2. Die abgerufenen Texte zusammenfassen (Strategie wird intern gewählt)
    if retrieved_texts:
        print(f"\nFasse {len(retrieved_texts)} abgerufene Dokumente zusammen...")
        # max_tokens für die *finale* Zusammenfassung anpassen
        summary = summarizer.summarize_texts(retrieved_texts, max_tokens=4096)
        print("\n--- Generierte Zusammenfassung ---")
        print(summary)
        print("------------------------------")
    else:
        print("\nKeine Texte in der Collection gefunden oder Fehler beim Abrufen.")
