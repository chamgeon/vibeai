from prompts import YOUTUBE_COMMENTS_DIGESTION_PROMPT
from scrapers import youtube_comment_scrape
from openai import OpenAI
import os
import time

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY']
)

def call_gpt(prompt, model, temperature = 0.9):

    if model == "gpt-5":
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={ "effort": "low"},
            text = {"verbosity": "medium"}
        )

        return response.output_text
    
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            temperature=temperature,
            max_tokens=1000
        )

        return response.choices[0].message.content


def youtube_comment_digest(song, artist, model = "gpt-5", max_comments = 25):

    """
    Generate vibe summary data from youtube comment.
    
    Input:
        song(str): song name to scrape
        artist(str): artist name to scrape
    
    Output:
        ChatGPT response that summarizes vibe of the song
    """

    youtube_comments = youtube_comment_scrape(song_name=song, artist_name=artist, max_comments = max_comments)
    raw_comments = ""
    for comment in youtube_comments:
        raw_comments += f"{comment.text}\n\n-----\n\n"
    
    print(raw_comments)
    
    prompt = YOUTUBE_COMMENTS_DIGESTION_PROMPT.replace("[COMMENTS]", raw_comments)
    vibe_digestion = call_gpt(prompt, model)

    return vibe_digestion




if __name__ == "__main__":
    song = 'bonfire'
    artist = 'wave to earth'
    digestion_data = youtube_comment_digest(song, artist)
    print(digestion_data)