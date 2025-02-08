from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
pipe = pipeline("text-generation", model="openai-community/gpt2")

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
        
    prompt = update.message.text
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    await update.message.reply_text(gen_text)

def main():
    # Replace with your actual token
    app = ApplicationBuilder().token("8078877052:AAFCvtm9-uMlncQLYgEl6Mk7F3G-L39fsDs").build()
    app.add_handler(CommandHandler("hello", hello))
    app.run_polling()

if __name__ == "__main__":
    main()
