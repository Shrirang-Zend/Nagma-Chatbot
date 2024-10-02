from nagma_chatbot import NagmaChatbot
from config import DATA_PATH

def main():
    chatbot = NagmaChatbot(DATA_PATH)
    chatbot.run()

if __name__ == "__main__":
    main()