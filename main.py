import urllib.request
from bs4 import BeautifulSoup
from requests import subjects_from_article

url = "https://www.forbes.com/sites/danidiplacido/2024/10/14/pokmon-fans-dont-understand-the-game-freak-leaks/"
url = "https://www.cnn.com/2016/08/02/politics/donald-trump-eats-kfc-knife-fork/index.html"
#url = "https://dailyhodl.com/2024/11/27/correction-for-bitcoin-in-coming-weeks-could-be-beneficial-for-bull-market-according-to-rekt-capital-heres-why/"

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
    article = get_text_from_url(url)
    if article is None:
        print("Could not extract text from URL")
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