from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("second_agent")
class SecondAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(SecondAgent, self).__init__()
        self.name = "SecondAgent"

    def step(self, chess_board, player, opponent):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (board_size, board_size)
         where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
         and 2 represents Player 2's discs (Brown).
        - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
        - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

        You should return a tuple (r,c), where (r,c) is the position where your agent
        wants to place the next disc. Use functions in helpers to determine valid moves
        and more helpful tools.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time

        print("My AI's turn took ", time_taken, "seconds.")

        # Get valid moves
        valid_moves = get_valid_moves(chess_board, player)

        # If no valid moves, return None
        if not valid_moves:
            return None

        # Use Minimax to find the best move
        best_move = None
        best_score = -float('inf')
        depth = 3  # Adjust the depth as needed
        for move in valid_moves:
            new_board = np.copy(chess_board)
            execute_move(new_board, move, player)
            score = self.minimax(new_board, depth - 1, -float('inf'), float('inf'), False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def minimax(self, board, depth, alpha, beta, is_maximizing_player):
        """
        Minimax search algorithm with alpha-beta pruning.

        Args:
            board: The current board state.
            depth: The current depth of the search.
            alpha: The alpha value for pruning
            beta: The beta value for pruning.
            is_maximizing_player: Whether the current player is maximizing or minimizing.

        Returns:
            The score of the best move.
        """

        if depth == 0 or check_endgame(board):
            return self.evaluate_board(board)

        if is_maximizing_player:
            best_value = -float('inf')
            for move in get_valid_moves(board, self.player):
                new_board = np.copy(board)
                execute_move(new_board, move, self.player)
                value = self.minimax(new_board, depth - 1, alpha, beta, False)
                best_value = max(best_value, value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return best_value
        else:
            best_value = float('inf')
            for move in get_valid_moves(board, self.opponent):
                new_board = np.copy(board)
                execute_move(new_board, move, self.opponent)
                value = self.minimax(new_board, depth - 1, alpha, beta, True)
                best_value = min(best_value, value)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return best_value

    def evaluate_board(self, board):
        """
        Evaluates the board state from the perspective of the second agent.

        Args:
            board: The current board state.

        Returns:
            The score of the board state.
        """

        player_score = np.sum(board == self.player)
        opponent_score = np.sum(board == self.opponent)

        # Basic score difference
        score = player_score - opponent_score

        # Corner occupancy bonus
        corner_squares = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for corner in corner_squares:
            if board[corner] == self.player:
                score += 4
            elif board[corner] == self.opponent:
                score -= 4

        # Edge and corner adjacency penalty
        edge_squares = [(0, 1), (0, 6), (1, 0), (1, 7), (6, 0), (6, 7), (7, 1), (7, 6)]
        for edge in edge_squares:
            if board[edge] == self.opponent:
                score -= 2

        # Mobility bonus
        player_moves = len(get_valid_moves(board, self.player))
        opponent_moves = len(get_valid_moves(board, self.opponent))
        score += player_moves - opponent_moves

        return score
