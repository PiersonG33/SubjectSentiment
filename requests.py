from openai import OpenAI

with open('apikey.txt', 'r') as f:
    key = f.read().strip()
client = OpenAI(api_key=key)

def subjects_from_article(article, max_subjects = 5):
    # Article will be a string of the full text of a news article
    # GPT should return a list of subjects that the article is about
    prompt = "Extract the {0} most important subjects from the following article. \
        Important subjects should be proper nouns specifically mentioned in the article. \
        Display only the subjects, separated by commas.\n{1}".format(max_subjects, article)
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    subjects = completion.choices[0].message.content
    return subjects