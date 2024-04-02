# setup

```
create a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
# game data

file structure: text file that contains one line per game
on each line are two integers between 0 and 2, separated by a space
the first integer is what i chose, the second is what the computer chose
0 = rock, 1 = paper, 2 = scissors
the last line contains the win record as wins, losses, and ties, each seperated by a space

sample
1 2
2 0
2 1
2 1 0

# execution

```
python main.py
```