import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class TextProcessor:
    def __init__(self, stemmer_language: str = 'english'):
        """
        Initializes the TextProcessor with the specified stemmer language.

        Args:
            stemmer_language (str): The language for the SnowballStemmer. Default is 'english'.
        """
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer(stemmer_language)

        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+', re.MULTILINE)
        self.ordinal_pattern = re.compile(r'\b\d+(st|nd|rd|th)\b')
        self.number_pattern = re.compile(r'\d+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.single_letter_pattern = re.compile(r'\b[a-zA-Z]\b')

    def apply_lower_case(self, text: str) -> str:
        """
        Converts text to lowercase and strips leading and trailing whitespace.

        Args:
            text (str): Input text.

        Returns:
            str: Lowercased and stripped text.
        """
        return text.lower().strip()

    def unify_whitespaces(self, text: str) -> str:
        """
        Replaces multiple whitespace characters with a single space.

        Args:
            text (str): Input text.

        Returns:
            str: Text with unified whitespaces.
        """
        return re.sub(r'\s+', ' ', text).strip()

    def remove_links(self, text: str) -> str:
        """
        Removes URLs from the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with URLs removed.
        """
        cleaned_text = self.url_pattern.sub('', text)
        return self.unify_whitespaces(cleaned_text)

    def remove_ordinals(self, text: str) -> str:
        """
        Removes ordinal numbers from the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with ordinal numbers removed.
        """
        cleaned_text = self.ordinal_pattern.sub('', text)
        return self.unify_whitespaces(cleaned_text)

    def remove_numbers(self, text: str) -> str:
        """
        Removes numerical digits from the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with numbers removed.
        """
        cleaned_text = self.number_pattern.sub('', text)
        return self.unify_whitespaces(cleaned_text)

    def remove_punctuation(self, text: str) -> str:
        """
        Removes punctuation characters from the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with punctuation removed.
        """
        cleaned_text = self.punctuation_pattern.sub('', text)
        return self.unify_whitespaces(cleaned_text)

    def remove_single_letters(self, text: str) -> str:
        """
        Removes single-letter words from the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with single-letter words removed.
        """
        cleaned_text = self.single_letter_pattern.sub('', text)
        return self.unify_whitespaces(cleaned_text)

    def remove_stopwords(self, text: str) -> str:
        """
        Removes common stopwords from the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with stopwords removed.
        """
        words = word_tokenize(text)
        filtered_words = [
            word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def apply_stemming(self, text: str) -> str:
        """
        Applies stemming to the text using the specified stemmer language.

        Args:
            text (str): Input text.

        Returns:
            str: Stemmed text.
        """
        word_tokens = word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in word_tokens]
        return ' '.join(stemmed_words)

    def process_text(self, text: str) -> str:
        """
        Cleans and processes the input text through a series of transformations.

        Args:
            text (str): Input text to clean.

        Returns:
            str: Processed text.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        text = self.apply_lower_case(text)
        text = self.remove_links(text)
        text = self.remove_numbers(text)
        text = self.remove_punctuation(text)
        text = self.remove_ordinals(text)
        text = self.remove_single_letters(text)
        text = self.remove_stopwords(text)
        text = self.apply_stemming(text)

        return text
