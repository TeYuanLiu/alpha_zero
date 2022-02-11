# alpha-zero

A simplified implementation of the [AlphaGo Zero paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ) for game TicTacToe, based on [Surag Nair](https://github.com/suragnair)'s reinforcement learning logic and [Evgeny Tyurin](https://github.com/evg-tyurin)'s TicTacToe game logic. Surag's and Evgeny's work can be found [here](https://github.com/suragnair/alpha-zero-general).

### File overview

- `game.py` defines the game base class `Game`, which is inherited by class `TicTacToe` in `tictactoe.py`. `net.py` defines the neural network base class `Net`, which is inherited by class `TicTacToeNet` in `tictactoenet.py`.

- `bot.py` defines the class `Bot`, which gets the game logic from `tictactoe.py` and the neural network from `tictactoenet.py` to do Monte Carlo tree search (MCTS) and find the best action given a game state.

- `incubator.py` contains the reinforcement learning logic for training and the battle logic for bot vs bot or bot vs human player.

- `main.py` demonstrates how to set parameters and conduct an end-to-end experiment including bot initialization, training, and evaluation.

- `app.py` shows my Flask app implementation to expose a pre-trained bot as an API. Given a game state query with `O` representing the bot and `X` the opponent, the API checks whether the game is still going and responds with the bot's action.

### Steps to get started

1.  Open a terminal and clone the repository.

    ```
    $ git clone https://github.com/Te-YuanLiu/alpha-zero.git
    ```

2.  Change directory into the cloned repository.

    ```
    $ cd alpha-zero/
    ```

3.  Create a python virtual environment and install all the required packages.

    ```
    $ python3 -m venv venv
    $ source ./venv/bin/activate
    $ pip3 install -r requirements.txt
    ```

4.  Customize `main.py` and run the full pipeline to train a TicTacToe bot from scratch and evaluate its performance by playing against it.

    ```
    $ vim main.py
    $ python3 main.py
    ```
