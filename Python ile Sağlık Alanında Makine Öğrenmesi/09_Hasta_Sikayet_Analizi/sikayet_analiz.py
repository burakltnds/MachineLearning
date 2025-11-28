# -*- coding: utf-8 -*-
"""
Updated on Fri Nov 28 2025
@author: burak
"""

from openai import OpenAI

client = OpenAI(api_key="")

def chat_with_gpt(prompt, history):
    

    messages = []

    for h in history:
        messages.append({"role": "user", "content": h})

    messages.append({
        "role": "user",
        "content": f"Sen bir hastane yöneticisisin. Yeni şikayet: {prompt}"
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    history = []
    
    while True:
        
        user_input = input("Mesajınız Nedir? ")

        if user_input.lower() in ["exit", ""]:
            print("Görüşme tamamlandı.")
            break
        
        history.append(user_input)

        response = chat_with_gpt(user_input, history)
        print("Chatbot:", response)
