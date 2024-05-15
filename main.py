import os
import math
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import subprocess
import customtkinter as ctk
from tkinter import filedialog
from CTkListbox import *
global selected_option


def data1():
    save_label = ctk.CTkLabel(root, text="Enter text to save", font=ctk.CTkFont(size=16))
    save_label.place(relx=0.14, rely=0.04)
    textbox_widget = ctk.CTkTextbox(root, height=400, width=350)
    textbox_widget.place(relx=0.05, rely=0.08)
    save_button = ctk.CTkButton(root, text="Save As", command=lambda: export_file())
    save_button.place(relx=0.14, rely=0.68)

    def export_file():
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    file_content = textbox_widget.get('1.0', ctk.END)
                    file.write(file_content)
                    save_label_message = ctk.CTkLabel(root, text="Saved", font=ctk.CTkFont(size=16))
                    save_label_message.place(relx=0.18, rely=0.74)
                    txt_files_listbox_func()
            except Exception as e:
                save_label_message = ctk.CTkLabel(root, text="ERROR", font=ctk.CTkFont(size=16))
                save_label_message.place(relx=0.18, rely=0.74)

    def txt_files_listbox_func():
        global txt_files_listbox, open_button
        try:
            txt_files_listbox.destroy()
            open_button.destroy()
        except:
            pass

        txt_files_listbox = CTkListbox(root, height=380, command=txt_file_open)
        txt_files_listbox.place(relx=0.38, rely=0.08)
        counter = 0
        for item in os.listdir("txt_files"):
            txt_files_listbox.insert(counter, item)
            counter += 1

        open_button = ctk.CTkButton(root, text="Open File", command=lambda: txt_file_open)
        open_button.place(relx=0.38, rely=0.68)

    def txt_file_open(selected_option):
        subprocess.call(['notepad.exe', 'txt_files/{}'.format(selected_option)])


    txt_files_listbox_func()


root = ctk.CTk()
root.title("Text Summarization")
root.geometry("1200x700")
root.resizable(width=False, height=False)

data1()

root.mainloop()

'''
def _create_frequency_matrix(sentences) -> dict:
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


def _create_tf_matrix(freq_matrix) -> dict:
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def _create_documents_per_words(freq_matrix) -> dict:
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents) -> dict:
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix) -> dict:
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


def _find_average_score(sentenceValue) -> float:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold) -> str:
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

with open("txt_files/metin2.txt", "r", newline="") as f:
    text = f.read()

# text = """Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.
# The standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested. Sections 1.10.32 and 1.10.33 from "de Finibus Bonorum et Malorum" by Cicero are also reproduced in their exact original form, accompanied by English versions from the 1914 translation by H. Rackham."""
# 1 Sentence Tokenize
sentences = sent_tokenize(text)
total_documents = len(sentences)
# print(sentences)

# 2 Create the Frequency matrix of the words in each sentence.
freq_matrix = _create_frequency_matrix(sentences)
# print(freq_matrix)


# 3 Calculate TermFrequency and generate a matrix
tf_matrix = _create_tf_matrix(freq_matrix)
# print(tf_matrix)

# 4 creating table for documents per words
count_doc_per_words = _create_documents_per_words(freq_matrix)
# print(count_doc_per_words)


# 5 Calculate IDF and generate a matrix
idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
# print(idf_matrix)

# 6 Calculate TF-IDF and generate a matrix
tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
# print(tf_idf_matrix)

# 7 Important Algorithm: score the sentences
sentence_scores = _score_sentences(tf_idf_matrix)
# print(sentence_scores)

# 8 Find the threshold
threshold = _find_average_score(sentence_scores)
# print(threshold)

# 9 Important Algorithm: Generate the summary
summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
print(summary)
'''
