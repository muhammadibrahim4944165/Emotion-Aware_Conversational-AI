from Speech_to_Speech import run_speech_to_speech
from Speech_to_Text import run_speech_to_text
from Text_to_Speech import run_text_to_speech

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1. Speech-to-Speech")
        print("2. Speech-to-Text")
        print("3. Text-to-Speech")
        print("4. Quit")
        print("Select the operation:")

        choice = input("Enter the number of your choice: ")

        if choice == '1':
            print("You selected the Speech-to-Speech model.")
            run_speech_to_speech()  # Use the function from speech_to_speech.py
        elif choice == '2':
            print("You selected the Speech-to-Text model.")
            run_speech_to_text()  # Use the function from speech_to_text.py
        elif choice == '3':
            print("You selected the Text-to-Speech model.")
            run_text_to_speech()  # Use the function from text_to_speech.py
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break  # Exit the loop and end the program
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main_menu()
