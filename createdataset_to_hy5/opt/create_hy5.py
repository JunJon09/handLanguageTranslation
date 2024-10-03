import os
class Movie():
    def __init__(self) -> None:
        self.directory_path =  "../../data/"
        self.word_list = []
        self.files_path_list = []

    def get_words_directory(self):
        try:
            files = os.listdir(self.directory_path)
            for file in files:
                if file == ".DS_Store":
                    continue
                self.word_list.append(file)
        except FileNotFoundError:
            print(f"ディレクトリ '{self.directory}' が存在しません。")
    
    def get_word_file_path(self):
        try:
            for word in self.word_list:
                full_path = os.path.join(self.directory_path, word)
                files = os.listdir(full_path)
                mp4_files = [os.path.join(full_path, f) for f in files if f !=".DS_Store" and f.lower().endswith('.mp4')]
                self.files_path_list.append(mp4_files)
        except FileNotFoundError:
            print(f"ファイルが存在しません。")

a = Movie()
a.get_words_directory()
a.get_word_file_path()