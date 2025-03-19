from vanir_parser.vanir_report_parser import VanirParser  
from download_diff.run_diff_fetcher import run_diff_fetcher

def main():
    filename = input("Input Vanir Output File Here: ")  
    try:
        VanirParser(filename)
        run_diff_fetcher()
    except FileNotFoundError:
        print("File not found, please ensure the path is correct.")
    except Exception as e: 
        print(f"An error occurred: {e}")

    kernel_path = input("Input Repository path to apply diff files here: ")  

if __name__ == "__main__":
    main()
