import urllib.request
from bs4 import BeautifulSoup
from requests import subjects_from_article
from sentence_processing import divide_by_subject
import sys
import sentiment
import gensim.downloader as api

url = ""
url = "https://www.forbes.com/sites/danidiplacido/2024/10/14/pokmon-fans-dont-understand-the-game-freak-leaks/"
# url = "https://www.cnn.com/2016/08/02/politics/donald-trump-eats-kfc-knife-fork/index.html"
# url = "https://dailyhodl.com/2024/11/27/correction-for-bitcoin-in-coming-weeks-could-be-beneficial-for-bull-market-according-to-rekt-capital-heres-why/"
#url = "https://www.yahoo.com/lifestyle/the-15-best-black-friday-deals-on-thanksgiving-day-2024-110031511.html"

def get_text_from_url(url):
    response = urllib.request.urlopen(url)
    html = response.read().strip()

    # dump the raw html to a file for debugging
    with open('raw.txt', 'wb') as f:
        f.write(html)

    soup = BeautifulSoup(html, 'html.parser')
    
    # Find the main content of the article
    article_body = soup.find('div', {'class': 'article-body'})
    if not article_body:
        article_body = soup.find('article')
    
    if not article_body:
        article_body = soup.find("div", class_="entry-content")
        if not article_body:
            return None

    # Remove figcaption tags
    for figcaption in article_body.find_all('figcaption'):
        figcaption.decompose()

    paragraphs = article_body.find_all('p')
    article = '\n'.join([p.get_text() for p in paragraphs])

    # Remove empty lines
    article = article.split('\n')
    article = [line.strip() for line in article if line.strip() != '']
    article = '\n'.join(article)

    # Replace HTML entities with their corresponding characters
    article = article.replace('&nbsp;', ' ').replace("&amp;", "&").replace("&quot;", '"').replace("&apos;", "'").replace("&lt;", "<").replace("&gt;", ">")
    article = article.replace('’', "'").replace('“', '"').replace('”', '"').replace("‘", "'")

    return article

def main():
    global url
    url = ""
    if url == "":
        args = sys.argv
        if len(args) < 2:
            url = input("Enter the URL of the article: ").strip()
        else:
            url = args[1].strip()


    article = get_text_from_url(url)
    if article is None:
        print("Could not extract text from URL")
        return
    else:
        with open('output.txt', 'w', encoding='utf-8') as f:
            f.write(article)
        print("Text extracted from URL and saved to output.txt")
    subjects = []
    subjects = subjects_from_article(article, "3")
    subjects = [s.strip() for s in subjects.split(',')]
    print(subjects)
    subdict = divide_by_subject(article, subjects)

    #Write to file for debugging
    # with open('output3.txt', 'w', encoding='utf-8') as f:
    #     for sub in subdict:
    #         f.write(sub + ":\n")
    #         for sentence in subdict[sub]:
    #             f.write(sentence + "\n")
    #         f.write("\n") 

    info = api.info()  # show info about available models/datasets
    model = api.load("glove-twitter-25")  # download the model and return as object ready for use

    sentiment_scores = []

    for x in subjects:
        print(x)
        print(subdict[x])
        sentiment_scores.append(sentiment.perform_sentiment_analysis_new(subdict[x], model))

    print(sentiment_scores)
    sentiment.plotting_new(sentiment_scores, subjects)

if __name__ == '__main__':
    main()