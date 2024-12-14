import yut.engine
import numpy as np

distance_to_goal = np.zeros(yut.rule.FINISHED + 1)

proximity_weight = 0.5  # Increased weight for positions closer to the goal

outcomes, probs = yut.rule.enumerate_all_cast_outcomes(depth=2)

for _ in range(10):
    for s in range(yut.rule.FINISHED):
        weighted_sum = 0.0
        for outcome, prob in zip(outcomes, probs):
            pos = s
            for ys in outcome:
                pos = yut.rule.next_position(pos, ys, True)

            # Base distance value
            base_value = 1 + distance_to_goal[pos]

            # Apply bonuses and penalties
            weighted_sum += base_value * prob
        distance_to_goal[s] = weighted_sum


class MyAlgo(yut.engine.Player):
    def __init__(self, max_depth=2): # Change depth does not really affect performance since it is mostly random dice outcomes

        self.max_depth = max_depth
        
    
    def name(self):
        return "Cool Minimax Player (202411)"
    
    def _evaluate_board_state(self, my_positions, enemy_positions, mal_caught):
        my_duplicates = [sum(np == p for np in my_positions) for p in my_positions]
        enemy_duplicates = [sum(np == p for np in enemy_positions) for p in enemy_positions]
        multipliers = [1, 1, 0.7, 0.4, 0.3]  # For evaluating based on the distance

        #vulnerable_map (based on statistics, gae and gol appear almost 70% of the time, making 2 or 3 tiles ahead of opponent's mal a risky poition)
        vulnerable_map = {0: [], 1: [], 2: [0], 3:[0, 1], 4: [1, 2], 5: [2, 3], 6:[3, 4], 7:[4], 8:[6], 9: [6,7], 10: [7,8], 
                          11: [], 12:[10], 13:[], 14:[5], 15:[10,11,5,13], 16: [13,14], 17:[14], 18:[8,9], 19:[9], 20:[18], 
                          21:[18,19], 22:[19,20,16], 23:[11,12], 24:[12,15], 25:[20,21,16,17], 26:[17,21,22], 27:[22,25], 28:[25,26], 29:[26,27], 30:[]}

        capture_bonus = 0.07  # Bonus for capturing an enemy's mal (Second priority after finishing the furthest mal first)
        landing_penalty = -0.01

        evaluation = 0

        for p, np in zip(my_positions, my_duplicates):
            multiplier = multipliers[np] if p != 0 else 1
            evaluation -= distance_to_goal[p] * multiplier
            
            # Bonus for capturing mal
            if p in enemy_positions:
                evaluation += capture_bonus
            # Penalty for high risk squre
            for enemy_pos in enemy_positions:
                if enemy_pos in vulnerable_map[p]:
                    evaluation += landing_penalty*len(vulnerable_map[p]) # add penalty proportional to the riskiness
        
        # Evaluate the enemy's positions
        for p, np in zip(enemy_positions, enemy_duplicates):
            multiplier = multipliers[np] if p != 0 else 1
            evaluation += distance_to_goal[p] * multiplier

            # Penalty for capturing mal
            if p in my_positions:
                evaluation -= capture_bonus
            # Bonus for high risk square
            for my_pos in my_positions:
                if my_pos in vulnerable_map[p]:
                    evaluation -= landing_penalty*len(vulnerable_map[p])

        return evaluation

    
    def _is_shortcut_possible(self, position):
        shortcut_positions = {15, 22, 29} 
        return position in shortcut_positions
    

    def minimax(self, my_positions, enemy_positions, available_yutscores, depth, alpha, beta, is_maximizing, mal_caught):

        # Base case
        if depth == 0:
            return self._evaluate_board_state(my_positions, enemy_positions, mal_caught), None

        current_positions = my_positions if is_maximizing else enemy_positions
        best_score = float('-inf') if is_maximizing else float('inf')
        best_move = None

        for mal_index, mal_pos in enumerate(current_positions):

            if mal_pos == yut.rule.FINISHED:
                continue

            for ys in available_yutscores:
                shortcuts = [True, False] if self._is_shortcut_possible(mal_pos) else [False]

                for shortcut in shortcuts:
                    # Simulate the move
                    if is_maximizing:
                        legal_move, next_my_positions, next_enemy_positions, mal_caught = yut.rule.make_move(
                            my_positions, enemy_positions, mal_index, ys, shortcut
                        )
                    else:
                        legal_move, next_enemy_positions, next_my_positions, mal_caught = yut.rule.make_move(
                            enemy_positions, my_positions, mal_index, ys, shortcut
                        )

                    if not legal_move:
                        continue

                    # Recursive call to minimax with Alpha-Beta Pruning
                    scores = []
                    outcomes, probs = yut.rule.enumerate_all_cast_outcomes(depth=1) # can be changed, depth 3 would be the best but it takes too long. Depth 1 and 2 show no significnt difference

                    for outcome, prob in zip(outcomes, probs):
                        for next_ys in outcome:
                            score, _ = self.minimax(
                                next_my_positions, 
                                next_enemy_positions, 
                                [next_ys], 
                                depth - 1, 
                                alpha, 
                                beta, 
                                not is_maximizing,
                                mal_caught
                            )

                            
                            scores.append(score)
                    score = np.max(scores) if is_maximizing else np.min(scores)

                    # Update best score and best move
                    if is_maximizing:
                        if score > best_score:
                            best_score = score
                            best_move = (mal_index, ys, shortcut)

                        # Alpha-Beta Pruning
                        alpha = max(alpha, best_score)
                        if alpha >= beta:
                            return best_score, best_move  # Prune branches
                    else:
                        if score < best_score:
                            best_score = score
                            best_move = (mal_index, ys, shortcut)

                        # Alpha-Beta Pruning
                        beta = min(beta, best_score)
                        if beta <= alpha:
                            return best_score, best_move  # Prune branches

        return best_score, best_move

    def action(self, state):
        _, my_positions, enemy_positions, available_yutscores = state
        # Run minimax with Alpha-Beta Pruning to find the best move
        _, best_move = self.minimax(
            my_positions, enemy_positions, available_yutscores, 
            depth=self.max_depth, 
            alpha=float('-inf'), 
            beta=float('inf'), 
            is_maximizing=True,
            mal_caught=0
        )

        # If no valid move found, return a default move
        if best_move is None:
            # Try to find any legal move
            for mal_index, mal_pos in enumerate(my_positions):
                if mal_pos == yut.rule.FINISHED:
                    continue
                for ys in available_yutscores:
                    for shortcut in [True, False]:
                        legal_move, _, _, _ = yut.rule.make_move(my_positions, enemy_positions, mal_index, ys, shortcut)
                        if legal_move:
                            return mal_index, ys, shortcut, ""

            # Absolute fallback
            return 0, available_yutscores[0], True, ""

        return best_move[0], best_move[1], best_move[2], ""
        
if __name__ == "__main__":
	p = MyAlgo()
	engine = yut.engine.GameEngine()
	for s in range(100):
		winner = engine.play( p, p, seed=s )
		if winner == 0:
			print( "Player 1 won!" )
		else:
			print( "Player 2 won!" )