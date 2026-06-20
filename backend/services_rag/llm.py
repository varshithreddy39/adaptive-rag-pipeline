from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API")
)


def generate_answer(
    query,
    chunks,
    history=None,
    summary=""
):

    context = "\n\n".join(chunks)

    conversation_history = ""

    if history:

        for chat in history:

            conversation_history += (
                f"User: {chat['user']}\n"
                f"Assistant: {chat['assistant']}\n\n"
            )

    prompt = f"""

You are a helpful AI assistant.

Answer the user's question ONLY using:
- the retrieved context
- the conversation summary
- the recent conversation

If the answer is not available in the retrieved context, say:

"I could not find the answer in the uploaded document."

================ CONVERSATION SUMMARY ================

{summary}

================ RECENT CONVERSATION ================

{conversation_history}

================ CONTEXT ================

{context}

================ QUESTION ================

{query}
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": "You answer questions from uploaded documents."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content