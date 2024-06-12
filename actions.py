import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import List, Dict

class ActionQueryGpt(Action):

    def name(self) -> str:
        return "action_query_gpt"

    def __init__(self):
        self.results_path = os.path.abspath(r"C:\Users\OLUWATOSIN OLATUNBOS\PycharmProjects\CU BOT\readpdf\results")
        print(f"Results path: {self.results_path}")
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.results_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.results_path)
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading model/tokenizer: {e}")

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List[Dict]:
        user_message = tracker.latest_message.get('text')
        try:
            inputs = self.tokenizer.encode(user_message + self.tokenizer.eos_token, return_tensors="pt")
            attention_mask = torch.ones(inputs.shape, device=inputs.device)

            print("Encoded input:", inputs)
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

            print("Model outputs:", outputs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Decoded response:", response)

            dispatcher.utter_message(text=response)
        except Exception as e:
            print(f"Error during run method: {e}")
            dispatcher.utter_message(text="Sorry, I couldn't process your request. Please try again later.")
        return []
