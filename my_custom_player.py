from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
        #From class MinimaxPlayer.sample_players.py:
        # randomly select a move as player 1 or 2 on an empty board, otherwise
        # return the optimal minimax move at a fixed search depth of 3 plies
        if state.ply_count < 2: self.queue.put(random.choice(state.actions()))

        depth_max = 2
        for d in range(1, depth_max + 1):
            self.queue.put(self.alfabeta(state, depth=d))

    def alfabeta(self, state, depth, alfa=float("-inf"), beta=float("inf")):

        alfa = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = state.actions()[0]

        for action in state.actions():
            value = self.min_value_ab(state.result(action), depth - 1, alfa, beta)
            alfa = max(alfa, value)
            if value > best_score:
                best_score = value
                best_move = action
        return best_move

    def min_value_ab(self, state, depth, alfa, beta):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth == 0: return self.score(state)
        v = float("inf")

        for action in state.actions():
            v = min(v, self.max_value_ab(state.result(action), depth - 1, alfa, beta))
            if v <= alfa:
                return v
            beta = min(beta, v)
            if beta <= alfa:
                break
        return v

    def max_value_ab(self, state, depth, alfa, beta):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth == 0: return self.score(state)
        v = float("-inf")

        for action in state.actions():
            v = max(v, self.min_value_ab(state.result(action), depth - 1, alfa, beta))
            if v >= beta:
                return v
            alfa = max(alfa, v)
            if alfa >= beta:
                break
        return v

    def score0(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    def  score1(self, state):
        """
        Custom Heuristic 1: Calculates the Manhattan distance from each of the
        players location to the center of the board, which is located on cell 57.
        If our player is closer to the center of the board than the opponent its
        got positional advantage as being closer to the walls decreases the
        options for moves.
        """
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)

        if own_loc == None:
            own_loc = 57
        elif opp_loc ==None:
            opp_loc = 57

        x_center, y_center = (7, 5) #The coordinates of the central cell (location 57) on the board
        x_own, y_own = (own_loc%13 + 2, int(own_loc/13) + 1) #This equation calculates the coordinates given the location
        x_opp, y_opp = (opp_loc%13 + 2, int(opp_loc/13) + 1)
        own_centerd = abs(x_own - x_center) + abs(y_own - y_center)
        opp_centerd = abs(x_opp - x_center) + abs(y_opp - y_center)
        manhattan_d = opp_centerd - own_centerd
        #print("manhattan: ", manhattan_d)
        return manhattan_d/10.0

    def  score(self, state):
        """
        Custom Heuristic 1: It first considers the default heuristic which returns the difference between the number of liberties
        of our player and the opponent. If the number of Liberties is the same the Heuristic calculates the Manhattan distance
        from each of the players location to the center of the board, which is located on cell 57. If our player is closer to the
        center of the board than the opponent its got positional advantage as being closer to the walls decreases the options for
        moves.
        """
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        if len(own_liberties) != len(opp_liberties):
            oldscore =  len(own_liberties) - 1.5*len(opp_liberties)
            #print("oldscore: ", oldscore)
            return oldscore
        elif len(own_liberties) == len(opp_liberties):
            if own_loc == None:
                own_loc = 57
            elif opp_loc ==None:
                opp_loc = 57
            x_center, y_center = (7, 5) #The coordinates of the central cell (location 57) on the board
            x_own, y_own = (own_loc%13 + 2, int(own_loc/13) + 1) #This equation calculates the coordinates given the location
            x_opp, y_opp = (opp_loc%13 + 2, int(opp_loc/13) + 1)
            own_centerd = abs(x_own - x_center) + abs(y_own - y_center)
            opp_centerd = abs(x_opp - x_center) + abs(y_opp - y_center)
            manhattan_d = opp_centerd - own_centerd
            #print("manhattan: ", manhattan_d)
            return manhattan_d/10.0
