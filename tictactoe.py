import random
import math

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'

    def print_board(self):
        for i in range(0, 9, 3):
            print(self.board[i], '|', self.board[i+1], '|', self.board[i+2])

    def check_winner(self):
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for combo in winning_combos:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != ' ':
                return self.board[combo[0]]
        return None

    def get_legal_moves(self):
        return [i for i, val in enumerate(self.board) if val == ' ']

    def make_move(self, move):
        self.board[move] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def simulate_game(self):
        while not self.check_winner() and self.get_legal_moves():
            move = random.choice(self.get_legal_moves())
            self.make_move(move)
        return self.check_winner()

    def monte_carlo_move(self, iterations=100000):
        scores = [0] * 9
        for _ in range(iterations):
            for move in self.get_legal_moves():
                board_copy = self.board.copy()
                player_copy = self.current_player
                self.make_move(move)
                winner = self.simulate_game()
                if winner == self.current_player:
                    scores[move] += 1
                elif winner is None:
                    scores[move] += 0.5
                self.board = board_copy
                self.current_player = player_copy
        return scores.index(max(scores))

if __name__ == "__main__":
    game = TicTacToe()
    while not game.check_winner() and game.get_legal_moves():
        game.print_board()
        if game.current_player == 'X':
            move = int(input("Enter your move (0-8): "))
        else:
            move = game.monte_carlo_move()
            print("AI's move:", move)
        game.make_move(move)
    game.print_board()
    if game.check_winner():
        print(game.check_winner(), "wins!")
    else:
        print("It's a draw!")