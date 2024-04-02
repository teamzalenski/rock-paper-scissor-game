from collections import Counter
import numpy as np
from scipy.special import gamma


class Game:
    def __init__(self):
        print("init")
        self.lookup = {"rock": 0, "paper": 1, "scissors": 2}
        self.lookup_reversed = {v: k for k, v in self.lookup.items()}
        self.games_to_play = 0

    # function that returns which one of two moves won
    # return -1 is round is a draw
    def get_winner(self, move1, move2):
        if (move1 == move2):
            return -1
        if (move1 + 1) % 3 == move2:
            return move2
        return move1

    # define your algorithm here or define your own functions and procedure
    def rock_paper_scissor(self, history=None):
        # define your function here
        pass

    # taken from the Dirichlet_Distribution_Visualization lab
    # create the matrices for mu1-3
    def create_mu_matrices(self, N_matrix_dim):
        # create a mesh grid for x and y
        x = np.linspace(0, 1, N_matrix_dim)
        y = np.linspace(0, 1, N_matrix_dim)
        mu_x_temp, mu_y_temp = np.meshgrid(x, y)
        # crate the grid for z implicitly, fixing rounding errors as we go    
        mu_z_temp = np.round(1-mu_x_temp - mu_y_temp, 10)
        mu_z_temp = np.where(mu_z_temp == -0, 0, mu_z_temp)
        # constrain the matrices to triangles
        mu_x = np.where(mu_z_temp < 0, np.nan, mu_x_temp)
        mu_y = np.where(mu_z_temp < 0, np.nan, mu_y_temp)
        mu_z = np.where(mu_z_temp < 0, np.nan, mu_z_temp)
        return mu_x, mu_y, mu_z

    # based on Dirichlet_Distribution_Bayesian_Update lab
    def bayesian_update(self, posterior, mu_x, mu_y, mu_z, alphas,
                        new_data, i):
        # update the alphas with the new data
        alphas = alphas + new_data
        # compute dirichlet distribution for K=3
        numerator = gamma(np.sum(alphas))
        denominator = gamma(alphas[0]) * gamma(alphas[1]) * gamma(alphas[2])
        mus_raised = (mu_x ** (alphas[0] - 1)) * (mu_y ** (alphas[1] - 1)) * (mu_z ** (alphas[2] - 1))
        dirichlet_dist = (numerator / denominator) * mus_raised
        # we want to make sure that the function is only defined
        # in the same region as mu_x, mu_y, and mu_z:
        dirichlet_dist = np.where(np.isnan(mu_z), np.NaN, dirichlet_dist)
        # store it
        posterior[:, :, i] = dirichlet_dist
        # move the iterator forward
        i += 1
        return posterior, alphas, i

    # function that calculates mu1-3 and returns the best move best
    # on which is biggest
    def generate_next_move(self, posterior, mu_x, mu_y, i):
        # find the max value in that slice
        flattened_index = np.nanargmax(posterior[:, :, i - 1])
        # map the flattened index to a coordinate
        x, y = np.unravel_index(flattened_index, posterior[:, :, i - 1].shape)
        # create a data structure for the mu values
        mu = np.zeros(3)
        # grab the mus for rock and paper
        mu[0] = mu_x[x][y]
        mu[1] = mu_y[x][y]
        # calculate the mu for scissors
        mu[2] = 1 - mu[0] - mu[1]
        # and finally pick the winner
        return np.argmax(mu)

    # function that takes in a new game and generates new data from it
    # x is 0-2 and represents me for rock-scissors, y is the same for
    # the computer
    # return a new data array of 0s with one element set to 1that can be
    # added in a bayesian iteration
    def parse_game(self, x, y):
        data = np.zeros(3)
        data[self.get_winner(x, y)] = 1
        return data

    def start(self):
        # read in the file, convert each line into a tuple
        with open("felix_data.txt", "r") as f:
            games = []
            lines = f.readlines()
            for line in lines[:-1]:
                # last line is the record, skip for processing games
                games.append(tuple(map(int, line.strip().split())))
            # process last line
            record = tuple(map(lambda x: int, lines[-1].strip().split()))
        # map each game to the winner. draws don't do any good, so those will be discounted eventually
        counts = Counter(map(lambda x: self.get_winner(*x), games))
        # figure out how many slices we need for the posterior
        self.games_to_play = sum(counts.values())
        # set up data structures based on
        # Dirichlet_Distribution_Bayesian_Update
        # set matrix dimensions for mu
        N_matrix_dim = 128
        # generate the meshgrid for mu_x, mu_y, and mu_z
        mu_x, mu_y, mu_z = self.create_mu_matrices(N_matrix_dim)
        # create data structure for alphas
        alphas = np.zeros(3)
        # create data structure for the posterior
        posterior = np.zeros([len(mu_x), len(mu_x), self.games_to_play])
        # pull out the alphas for each K. be conservative, 
        # each K should be present, but who knows
        prior = np.zeros(3)
        for k in range(3):
            if k in counts:
                prior[k] = counts[k]
        # we're on iteration 0
        i = 0
        # initialize with the prior
        posterior, alphas, i = self.bayesian_update(posterior, mu_x, mu_y, mu_z, alphas, prior, i)        
        # Run your algorithm to play against the computer
        my_move = self.generate_next_move(posterior, mu_x, mu_y, i)
        print(f"You should play {self.lookup_reversed[my_move]}")
        for game in range(self.games_to_play):
            # ask for what the computer played
            their_move = input("What did the computer play? 0 = rock, 1 = paper, 2 = scissors: ")
            # ties don't change things
            if their_move != my_move:
                new_data = self.parse_game(int(my_move), int(their_move))
                posterior, alphas, i = self.bayesian_update(posterior, 
                                                            mu_x, mu_y, mu_z, 
                                                            alphas, new_data, i)
        my_move = self.generate_next_move(posterior, mu_x, mu_y, i)
        print(f"{self.games_to_play - game - 1} games left. You should play {self.lookup_reversed[my_move]}")    


if __name__ == "__main__":
    game = Game()
    game.start()
