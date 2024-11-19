import urllib.request
from bs4 import BeautifulSoup
from requests import subjects_from_article

url = "https://www.forbes.com/sites/danidiplacido/2024/10/14/pokmon-fans-dont-understand-the-game-freak-leaks/"
#url = "https://www.cnn.com/2016/08/02/politics/donald-trump-eats-kfc-knife-fork/index.html"

def get_text_from_url(url):
        # Testing text extraction from URL
    try:
        response = urllib.request.urlopen(url)
        html = response.read().strip()

        # Isolate the news article text
        soup = BeautifulSoup(html, 'html.parser')
        article = soup.find('article').get_text()
        

        #Remove empty lines:
        article = article.split('\n')
        article = [line.strip() for line in article if line.strip() != '']
        article = '\n'.join(article)

        # Replace HTML entities with their corresponding characters
        print("Starting to replace HTML entities")
        article = article.replace('&nbsp;', ' ').replace("&amp;", "&").replace("&quot;", '"').replace("&apos;", "'").replace("&lt;", "<").replace("&gt;", ">")
        article = article.replace('’', "'").replace('“', '"').replace('”', '"').replace("‘", "'")

        
        # Output raw HTML to file:
        # with open('raw.txt', 'w', encoding='utf-8') as f:
        #     f.write(str(soup))

        return article
    except:
        print("Error while trying to access the URL")
        return None

def main():
    article = get_text_from_url(url)
    if article is None:
        return
    else:
        with open('output.txt', 'w', encoding='utf-8') as f:
            f.write(article)
        print("Text extracted from URL and saved to output.txt")
    subjects = subjects_from_article(article, 5)
    subjects = [s.strip() for s in subjects.split(',')]
    print(subjects)

if __name__ == '__main__':
    main()