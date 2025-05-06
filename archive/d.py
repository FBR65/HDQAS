from openai import OpenAI
import os

# Configure OpenAI to use Ollama's API
api_base = "http://localhost:11434/v1"  # Replace with your Ollama API endpoint
api_key = os.environ.get(
    "OPENAI_API_KEY", "ollama"
)  # Or set it directly if not using env vars

# Create an OpenAI client instance
client = OpenAI(
    api_key=api_key,
    base_url=api_base,
)


def ollama_coreference_resolution(text):
    """
    Performs coreference resolution on German text using Ollama and Qwen2.5
    through the OpenAI-compatible API.
    """
    try:
        # Define the system prompt
        system_prompt = """Löse die Koreferenz in den folgenden Textabschnitten auf und gib die Ergebnisse in derselben Reihenfolge wie die Eingabetexte aus. Füge nichts hinzu und lasse nichts weg.
        Du sollst ausschließlich die Konvertierung ausgeben, nicht mehr.

Beispiele:
"John ist ein begeisterter Radfahrer. John liebt es, mit seinem Mountainbike neue Wege zu erkunden. Letztes Wochenende unternahm John eine anspruchsvolle Tour durch das hügelige Gelände."
"Lisa besuchte ihre Großmutter. Sie brachte ihr einen Blumenstrauß mit."
"Michael hat einen neuen Hund. Er ist sehr verspielt und liebt es, im Park zu spielen."
"Sarah liebt ihren Bruder sehr. Er ist ihr bester Freund."
"David kaufte ein neues Auto. Es ist sehr geräumig und hat viele Extras."
"Emily hilft ihrer Mutter oft im Haushalt. Sie mag es, gemeinsam zu kochen."

Antwort:
"John ist ein begeisterter Radfahrer. John liebt es, mit seinem Mountainbike neue Wege zu erkunden. Letztes Wochenende unternahm John eine anspruchsvolle Tour durch das hügelige Gelände.","Lisa besuchte ihre Großmutter. Lisa brachte der Großmutter einen Blumenstrauß mit."
"Michael hat einen neuen Hund. Der Hund ist sehr verspielt und liebt es, im Park zu spielen."
"Sarah liebt ihren Bruder sehr. Ihr Bruder ist ihr bester Freund."
"David kauft ein neues Auto. Das neue Auto ist sehr geräumig und hat viele Extras."
"Emily hilft ihrer Mutter oft im Haushalt. Sie und ihre Mutter mag es, gemeinsam zu kochen."""

        # Define the user prompt (the text to be processed)
        user_prompt = text

        # Generate the rewritten text using OpenAI (Ollama)
        completion = client.chat.completions.create(
            model="qwen2.5",  # Replace with the modeclientl you want to use
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        rewritten_text = completion.choices[0].message.content

        return rewritten_text

    except Exception as e:
        print(f"An error occurred: {e}")
        return text


if __name__ == "__main__":
    test_cases = [
        "Lisa besuchte ihre Großmutter. Sie brachte ihr einen Blumenstrauß mit.",
        "Michael hat einen neuen Hund. Er ist sehr verspielt und liebt es mit ihm im Park zu spielen.",
        "Sarah liebt ihren Bruder sehr. Er ist ihr bester Freund.",
        "David kaufte ein neues Auto. Es ist sehr geräumig und hat viele Extras.",
        "Emily hilft ihrer Mutter oft im Haushalt. Sie mag es, gemeinsam zu kochen.",
    ]

    for text in test_cases:
        resolved_text = ollama_coreference_resolution(text)
        print("Original Text:", text)
        print("Resolved Text:", resolved_text)
        print("-" * 20)
