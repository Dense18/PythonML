def run_command(command: str):
    match command:
        case "hi":
            print("hi")
        case "kek":
            print("kek")
        case woohoo:
            print(woohoo)
        
def run_command(command: str):
    match command.split():
        case [heh, sda, sasd]:
            print(f"cool, {heh}")
        case ["kek", heh]:
            print(f"kek, {heh}")
        case woohoo:
            print(woohoo)
            
def main():
    while True:
        command = input("$ ")
        run_command(command)

if __name__ == '__main__':
    main()