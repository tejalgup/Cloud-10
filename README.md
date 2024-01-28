## Inspiration
I've always loved rhythm games. Some of the more interactive ones require a lot of equipment and aren't easily accessible for those suffering from physical hardships. I created Cloud 10, an inclusive game ,meant to get you dancing!
## What it does
You pick a sing from the menu, and get ready... to start dancing! A video will play, and it's your job to try and dance along as best as possible.
## How we built it
I used mediapipe and opencv to create landmarks on the video as well as take a video input of the user dancing and map their body in real-time. The program has a GUI built from PyQT5 which encompasses a full menu with different song selections, the game itself, as well as a statistical tracker which analyzes how well the player is doing throughout the song and gives a final average rating.
## Challenges we ran into
Working with the framerate and hardware available to me bottlenecked the game unfortunately. The game was running in lower FPS and played through the videos slower than anticipated. This was worked around by adjusting the quality and speeding the videos up.
## Accomplishments that we're proud of
I'm proud I was able to get a pretty holistic game completed in the time slot. There are a lot of features in the program for just a day of hacking!
## What we learned
I haven't worked with GUI before, and I spent a lot of time learning to be able to create the interface all in Python (it was very difficult for me to get in to but rewarding to learn) I as well created some of the art to go along with the whole "theme" of the program so the user really feels like they're on Cloud 10.
## What's next for Cloud 10
Next, I'd love to be able to make a leaderboard to store data as well as improve the progress data tracker to make it more UI friendly!
